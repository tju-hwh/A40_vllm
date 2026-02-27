from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _strip_slash(url: str) -> str:
    return url.rstrip("/")


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class ProxyConfig:
    role: str
    primary_upstream: str
    alt_upstreams: list[str]
    routing_mode: str
    cutover_requests: int
    sequential_block_size: int
    sequential_targets: list[str]
    sequential_target_kv_ports: list[int]
    sequential_decode_tokens: list[int]
    request_timeout_s: float
    connect_timeout_s: float
    verbose_log: bool
    require_kv_transfer: bool

    @staticmethod
    def from_env() -> "ProxyConfig":
        role = os.getenv("PROXY_ROLE", "relay").strip().lower()
        if role not in {"ingress", "relay"}:
            raise ValueError(f"Unsupported PROXY_ROLE={role!r}, expected ingress|relay")

        primary = os.getenv("PRIMARY_UPSTREAM", "").strip()
        if not primary:
            raise ValueError("PRIMARY_UPSTREAM is required")

        alt_raw = os.getenv("ALT_UPSTREAMS", "").strip()
        alt = [_strip_slash(x.strip()) for x in alt_raw.split(",") if x.strip()]

        if role == "ingress" and not alt:
            raise ValueError("ALT_UPSTREAMS is required for ingress role")

        routing_mode = os.getenv("ROUTING_MODE", "cutover_rr").strip().lower()
        if routing_mode not in {"cutover_rr", "sequential_blocks", "sequential_handoff"}:
            raise ValueError(
                f"Unsupported ROUTING_MODE={routing_mode!r}, expected cutover_rr|sequential_blocks|sequential_handoff"
            )

        block_size = int(os.getenv("SEQUENTIAL_BLOCK_SIZE", "128"))
        if block_size <= 0:
            raise ValueError(f"SEQUENTIAL_BLOCK_SIZE must be > 0, got {block_size}")

        seq_targets_raw = os.getenv("SEQUENTIAL_TARGETS", "").strip()
        seq_targets = [_strip_slash(x.strip()) for x in seq_targets_raw.split(",") if x.strip()]
        if not seq_targets:
            seq_targets = [_strip_slash(primary), *alt]
        if role == "ingress" and routing_mode in {"sequential_blocks", "sequential_handoff"} and not seq_targets:
            raise ValueError("SEQUENTIAL_TARGETS resolved empty for ingress sequential_blocks mode")

        kv_ports_raw = os.getenv("SEQUENTIAL_TARGET_KV_PORTS", "").strip()
        kv_ports = [int(x.strip()) for x in kv_ports_raw.split(",") if x.strip()] if kv_ports_raw else []
        if kv_ports and len(kv_ports) != len(seq_targets):
            raise ValueError(
                f"SEQUENTIAL_TARGET_KV_PORTS size ({len(kv_ports)}) must equal SEQUENTIAL_TARGETS size ({len(seq_targets)})"
            )
        if not kv_ports:
            # Fallback to HTTP ports if KV ports are not explicitly provided.
            # This is not suitable for P2pNcclConnector, but keeps old behavior.
            kv_ports = [_target_host_port(t)[1] for t in seq_targets]

        seq_decode_tokens_raw = os.getenv("SEQUENTIAL_DECODE_TOKENS", "").strip()
        seq_decode_tokens = [
            int(x.strip()) for x in seq_decode_tokens_raw.split(",") if x.strip()
        ] if seq_decode_tokens_raw else [1000, 1000, 1000]
        if any(x <= 0 for x in seq_decode_tokens):
            raise ValueError(
                f"SEQUENTIAL_DECODE_TOKENS must be positive integers, got {seq_decode_tokens!r}"
            )

        return ProxyConfig(
            role=role,
            primary_upstream=_strip_slash(primary),
            alt_upstreams=alt,
            routing_mode=routing_mode,
            cutover_requests=int(os.getenv("CUTOVER_REQUESTS", "1000")),
            sequential_block_size=block_size,
            sequential_targets=seq_targets,
            sequential_target_kv_ports=kv_ports,
            sequential_decode_tokens=seq_decode_tokens,
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "300")),
            connect_timeout_s=float(os.getenv("CONNECT_TIMEOUT_S", "30")),
            verbose_log=_parse_bool(os.getenv("PROXY_VERBOSE_LOG"), default=False),
            require_kv_transfer=_parse_bool(os.getenv("REQUIRE_KV_TRANSFER"), default=False),
        )


class RouteState:
    def __init__(self, cfg: ProxyConfig):
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self.decode_request_count = 0
        self.post_cutover_rr_index = 0
        self.decode_target_counts: dict[str, int] = {}

    async def choose_target(self, path: str) -> tuple[str, Optional[int], str]:
        """Return target upstream, decode_idx (if counted), and route label."""
        is_decode_path = path in {"/v1/completions", "/v1/chat/completions"}
        if self.cfg.role == "relay":
            return self.cfg.primary_upstream, None, "relay_to_primary"

        if not is_decode_path:
            return self.cfg.primary_upstream, None, "ingress_passthrough_primary"

        async with self._lock:
            self.decode_request_count += 1
            decode_idx = self.decode_request_count
            if self.cfg.routing_mode == "sequential_handoff":
                return self.cfg.sequential_targets[0], decode_idx, "sequential_handoff_chain"
            if self.cfg.routing_mode == "sequential_blocks":
                block_idx = (decode_idx - 1) // self.cfg.sequential_block_size
                block_idx = min(block_idx, len(self.cfg.sequential_targets) - 1)
                target = self.cfg.sequential_targets[block_idx]
                self.decode_target_counts[target] = self.decode_target_counts.get(target, 0) + 1
                route_label = f"sequential_block_{block_idx + 1}"
                return target, decode_idx, route_label

            if decode_idx <= self.cfg.cutover_requests:
                self.decode_target_counts[self.cfg.primary_upstream] = (
                    self.decode_target_counts.get(self.cfg.primary_upstream, 0) + 1
                )
                return self.cfg.primary_upstream, decode_idx, "pre_cutover_primary"

            target = self.cfg.alt_upstreams[self.post_cutover_rr_index % len(self.cfg.alt_upstreams)]
            self.post_cutover_rr_index += 1
            self.decode_target_counts[target] = self.decode_target_counts.get(target, 0) + 1
            return target, decode_idx, "post_cutover_alt_rr"

    async def add_target_hit(self, target: str) -> None:
        async with self._lock:
            self.decode_target_counts[target] = self.decode_target_counts.get(target, 0) + 1


def _is_stream_request(body: bytes, content_type: str | None) -> bool:
    if not body:
        return False
    if not content_type or "application/json" not in content_type.lower():
        return False
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return False
    return bool(obj.get("stream", False))


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    banned = {"content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in banned}


def _as_completion_text(obj: dict[str, Any]) -> str:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("upstream completion response has no choices")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise ValueError("upstream completion response choices[0].text is missing")
    return text


def _build_handoff_plan(total_max_tokens: int,
                        targets: list[str],
                        cutovers: list[int]) -> list[tuple[str, int]]:
    if total_max_tokens <= 0:
        return []
    if not targets:
        return []

    remaining = total_max_tokens
    plan: list[tuple[str, int]] = []
    for idx, limit in enumerate(cutovers):
        if idx >= len(targets) - 1 or remaining <= 0:
            break
        hop_tokens = min(remaining, limit)
        if hop_tokens > 0:
            plan.append((targets[idx], hop_tokens))
            remaining -= hop_tokens
    if remaining > 0:
        last_idx = min(len(cutovers), len(targets) - 1)
        plan.append((targets[last_idx], remaining))
    return plan


def _target_host_port(target_base: str) -> tuple[str, int]:
    parsed = urlparse(target_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, int(port)


def _build_p2p_request_id(base_request_id: str, prev_target: str | None,
                          prev_kv_port: int | None, next_target: str | None,
                          next_kv_port: int | None) -> str:
    rid = base_request_id
    if prev_target:
        phost, _ = _target_host_port(prev_target)
        rid += f"___prefill_addr_{phost}:{int(prev_kv_port)}___"
    if next_target:
        dhost, _ = _target_host_port(next_target)
        rid += f"___decode_addr_{dhost}:{int(next_kv_port)}"
    return rid


async def _handle_completion_sequential_handoff(
    *,
    client: httpx.AsyncClient,
    cfg: ProxyConfig,
    state: RouteState,
    body: bytes,
    req_headers: dict[str, str],
    full_path: str,
    query: str,
    decode_idx: Optional[int],
) -> Response:
    try:
        req_obj = json.loads(body.decode("utf-8"))
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid json body: {exc}"})

    if req_obj.get("stream", False):
        return JSONResponse(
            status_code=400,
            content={"error": "stream=true is not supported in sequential_handoff mode"},
        )
    prompt = req_obj.get("prompt")
    if not isinstance(prompt, str):
        return JSONResponse(
            status_code=400,
            content={"error": "sequential_handoff currently supports string prompt only"},
        )

    total_max_tokens = int(req_obj.get("max_tokens", 16))
    if total_max_tokens <= 0:
        return JSONResponse(status_code=400, content={"error": "max_tokens must be > 0"})

    plan = _build_handoff_plan(
        total_max_tokens=total_max_tokens,
        targets=cfg.sequential_targets,
        cutovers=cfg.sequential_decode_tokens,
    )
    if not plan:
        return JSONResponse(status_code=400, content={"error": "empty handoff execution plan"})

    generated_text = ""
    current_prompt = prompt
    req_kv = req_obj.get("kv_transfer_params")
    kv_transfer_params: Optional[dict[str, Any]]
    if isinstance(req_kv, dict):
        kv_transfer_params = dict(req_kv)
    else:
        # Default to local-only; enable remote transfer per-hop below.
        kv_transfer_params = {"do_remote_prefill": False, "do_remote_decode": False}

    sum_completion_tokens = 0
    first_prompt_tokens: Optional[int] = None
    last_resp: dict[str, Any] | None = None
    first_id = req_obj.get("request_id")
    base_request_id = str(first_id) if first_id else f"handoff-{uuid.uuid4().hex}"

    for hop_idx, (target_base, hop_max_tokens) in enumerate(plan, start=1):
        is_last_hop = hop_idx == len(plan)
        await state.add_target_hit(target_base)
        prev_target = plan[hop_idx - 2][0] if hop_idx > 1 else None
        next_target = plan[hop_idx][0] if hop_idx < len(plan) else None
        target_index = cfg.sequential_targets.index(target_base)
        prev_kv_port = cfg.sequential_target_kv_ports[target_index - 1] if hop_idx > 1 else None
        next_kv_port = cfg.sequential_target_kv_ports[target_index + 1] if hop_idx < len(plan) else None

        hop_req = dict(req_obj)
        hop_req["stream"] = False
        hop_req["prompt"] = current_prompt
        hop_req["max_tokens"] = hop_max_tokens
        # Keep a stable logical request id and embed prefill/decode addrs for
        # connectors (notably P2pNcclConnector) to resolve recv/send peers.
        hop_req["request_id"] = _build_p2p_request_id(
            base_request_id,
            prev_target,
            prev_kv_port,
            next_target,
            next_kv_port,
        )
        if cfg.verbose_log:
            print(
                f"[proxy:{cfg.role}] handoff decode_idx={decode_idx} "
                f"hop={hop_idx}/{len(plan)} target={target_base} "
                f"max_tokens={hop_max_tokens} req_id={hop_req['request_id']}"
            )
        if kv_transfer_params is not None:
            hop_kv = dict(kv_transfer_params)
            # Only non-first hops need remote prefill (load previous KV).
            hop_kv["do_remote_prefill"] = hop_idx > 1
            # Only non-last hops need remote decode (export KV to next hop).
            hop_kv["do_remote_decode"] = not is_last_hop
            # Avoid sending kv_transfer_params for local-only hops. Some
            # connectors still enter KV path when this field exists.
            if hop_kv["do_remote_prefill"] or hop_kv["do_remote_decode"]:
                hop_req["kv_transfer_params"] = hop_kv

        upstream_url = f"{target_base}{full_path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"
        try:
            resp = await client.request(
                method="POST",
                url=upstream_url,
                headers=req_headers,
                content=json.dumps(hop_req).encode("utf-8"),
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "bad_gateway",
                    "detail": str(exc),
                    "upstream": target_base,
                    "path": full_path,
                    "route_label": "sequential_handoff_chain",
                    "decode_idx": decode_idx,
                    "hop": hop_idx,
                },
            )

        if resp.status_code >= 400:
            try:
                err_obj = resp.json()
            except Exception:
                err_obj = {"detail": resp.text}
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": "upstream_error",
                    "upstream": target_base,
                    "hop": hop_idx,
                    "detail": err_obj,
                    "decode_idx": decode_idx,
                },
            )

        try:
            resp_obj = resp.json()
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "invalid_upstream_json",
                    "upstream": target_base,
                    "hop": hop_idx,
                    "detail": str(exc),
                    "decode_idx": decode_idx,
                },
            )

        hop_text = _as_completion_text(resp_obj)
        if cfg.verbose_log:
            print(
                f"[proxy:{cfg.role}] handoff decode_idx={decode_idx} "
                f"hop={hop_idx}/{len(plan)} done status={resp.status_code} "
                f"text_len={len(hop_text)}"
            )
        generated_text += hop_text
        current_prompt += hop_text
        next_kv_transfer_params = resp_obj.get("kv_transfer_params")
        if not is_last_hop and cfg.require_kv_transfer and not isinstance(next_kv_transfer_params, dict):
            return JSONResponse(
                status_code=502,
                content={
                    "error": "kv_transfer_unavailable",
                    "detail": (
                        "Upstream did not return kv_transfer_params during sequential_handoff. "
                        "KVConnector is likely not enabled on this server."
                    ),
                    "upstream": target_base,
                    "hop": hop_idx,
                    "decode_idx": decode_idx,
                },
            )
        if isinstance(next_kv_transfer_params, dict):
            kv_transfer_params = next_kv_transfer_params

        usage_obj = resp_obj.get("usage") or {}
        if first_prompt_tokens is None and isinstance(usage_obj.get("prompt_tokens"), int):
            first_prompt_tokens = int(usage_obj["prompt_tokens"])
        if isinstance(usage_obj.get("completion_tokens"), int):
            sum_completion_tokens += int(usage_obj["completion_tokens"])
        last_resp = resp_obj

    assert last_resp is not None
    if "choices" in last_resp and isinstance(last_resp["choices"], list) and last_resp["choices"]:
        last_resp["choices"][0]["text"] = generated_text
    usage = last_resp.get("usage")
    if isinstance(usage, dict):
        if first_prompt_tokens is not None:
            usage["prompt_tokens"] = first_prompt_tokens
        usage["completion_tokens"] = sum_completion_tokens
        if first_prompt_tokens is not None:
            usage["total_tokens"] = first_prompt_tokens + sum_completion_tokens
    last_resp["kv_transfer_params"] = kv_transfer_params
    return JSONResponse(status_code=200, content=last_resp)


def create_app() -> FastAPI:
    cfg = ProxyConfig.from_env()
    state = RouteState(cfg)
    app = FastAPI(title=f"vLLM Proxy ({cfg.role})", version="0.1.0")
    app.state.cfg = cfg
    app.state.route_state = state
    app.state.client = None

    @app.on_event("startup")
    async def _startup() -> None:
        timeout = httpx.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s)
        limits = httpx.Limits(max_keepalive_connections=2048, max_connections=4096)
        app.state.client = httpx.AsyncClient(timeout=timeout, limits=limits)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        client: httpx.AsyncClient = app.state.client
        if client is not None:
            await client.aclose()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "role": cfg.role}

    @app.get("/__proxy_state")
    async def proxy_state() -> dict:
        return {
            "role": cfg.role,
            "routing_mode": cfg.routing_mode,
            "primary_upstream": cfg.primary_upstream,
            "alt_upstreams": cfg.alt_upstreams,
            "cutover_requests": cfg.cutover_requests,
            "sequential_block_size": cfg.sequential_block_size,
            "sequential_targets": cfg.sequential_targets,
            "sequential_target_kv_ports": cfg.sequential_target_kv_ports,
            "sequential_decode_tokens": cfg.sequential_decode_tokens,
            "decode_request_count": state.decode_request_count,
            "decode_target_counts": state.decode_target_counts,
            "post_cutover_rr_index": state.post_cutover_rr_index,
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def passthrough(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        full_path = "/" + path
        target_base, decode_idx, route_label = await state.choose_target(full_path)
        upstream_url = f"{target_base}{full_path}"

        body = await request.body()
        req_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        query = request.url.query
        if query:
            upstream_url = f"{upstream_url}?{query}"

        is_stream = _is_stream_request(body, request.headers.get("content-type"))
        if cfg.routing_mode == "sequential_handoff" and full_path == "/v1/completions":
            return await _handle_completion_sequential_handoff(
                client=client,
                cfg=cfg,
                state=state,
                body=body,
                req_headers=req_headers,
                full_path=full_path,
                query=query,
                decode_idx=decode_idx,
            )

        if cfg.verbose_log and decode_idx is not None:
            print(
                f"[proxy:{cfg.role}] decode_idx={decode_idx} route={route_label} "
                f"target={target_base} path={full_path}"
            )

        try:
            if is_stream:
                req = client.build_request(
                    method=request.method,
                    url=upstream_url,
                    headers=req_headers,
                    content=body,
                )
                resp = await client.send(req, stream=True)

                async def _iter() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    _iter(),
                    status_code=resp.status_code,
                    headers=_filter_response_headers(resp.headers),
                    media_type=resp.headers.get("content-type"),
                )

            resp = await client.request(
                method=request.method,
                url=upstream_url,
                headers=req_headers,
                content=body,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=_filter_response_headers(resp.headers),
                media_type=resp.headers.get("content-type"),
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "bad_gateway",
                    "detail": str(exc),
                    "upstream": target_base,
                    "path": full_path,
                    "route_label": route_label,
                    "decode_idx": decode_idx,
                },
            )

    return app
