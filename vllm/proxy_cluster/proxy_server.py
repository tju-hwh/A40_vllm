from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

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
    request_timeout_s: float
    connect_timeout_s: float
    verbose_log: bool

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
        if routing_mode not in {"cutover_rr", "sequential_blocks"}:
            raise ValueError(
                f"Unsupported ROUTING_MODE={routing_mode!r}, expected cutover_rr|sequential_blocks"
            )

        block_size = int(os.getenv("SEQUENTIAL_BLOCK_SIZE", "128"))
        if block_size <= 0:
            raise ValueError(f"SEQUENTIAL_BLOCK_SIZE must be > 0, got {block_size}")

        seq_targets_raw = os.getenv("SEQUENTIAL_TARGETS", "").strip()
        seq_targets = [_strip_slash(x.strip()) for x in seq_targets_raw.split(",") if x.strip()]
        if not seq_targets:
            seq_targets = [_strip_slash(primary), *alt]
        if role == "ingress" and routing_mode == "sequential_blocks" and not seq_targets:
            raise ValueError("SEQUENTIAL_TARGETS resolved empty for ingress sequential_blocks mode")

        return ProxyConfig(
            role=role,
            primary_upstream=_strip_slash(primary),
            alt_upstreams=alt,
            routing_mode=routing_mode,
            cutover_requests=int(os.getenv("CUTOVER_REQUESTS", "1000")),
            sequential_block_size=block_size,
            sequential_targets=seq_targets,
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "300")),
            connect_timeout_s=float(os.getenv("CONNECT_TIMEOUT_S", "30")),
            verbose_log=_parse_bool(os.getenv("PROXY_VERBOSE_LOG"), default=False),
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
