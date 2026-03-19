from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)


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
    sequential_target_dp_sizes: list[int]
    sequential_target_groups: list[list[str]]
    sequential_target_kv_port_groups: list[list[int]]
    sequential_decode_tokens: list[int]
    request_timeout_s: float
    connect_timeout_s: float
    verbose_log: bool
    require_kv_transfer: bool
    dynamic_kv_control_path: str
    dynamic_kv_wait_timeout_s: float
    dynamic_kv_settle_s: float
    kv_owner_state_url: str
    kv_owner_state_strict: bool
    kv_handoff_serial_barrier: bool
    kv_handoff_min_layers: int
    kv_handoff_wait_timeout_s: float
    kv_handoff_stable_polls: int
    kv_handoff_soft_min_layers: int
    kv_handoff_global_phase_barrier: bool
    max_response_length: int
    upstream_max_model_len: int

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

        dp_sizes_raw = os.getenv("SEQUENTIAL_TARGET_DP_SIZES", "").strip()
        dp_sizes = [int(x.strip()) for x in dp_sizes_raw.split(",") if x.strip()] if dp_sizes_raw else []
        if dp_sizes and len(dp_sizes) != len(seq_targets):
            raise ValueError(
                f"SEQUENTIAL_TARGET_DP_SIZES size ({len(dp_sizes)}) must equal SEQUENTIAL_TARGETS size ({len(seq_targets)})"
            )
        if not dp_sizes:
            dp_sizes = [1] * len(seq_targets)
        if any(x <= 0 for x in dp_sizes):
            raise ValueError(
                f"SEQUENTIAL_TARGET_DP_SIZES must be positive integers, got {dp_sizes!r}"
            )

        seq_target_groups_raw = os.getenv("SEQUENTIAL_TARGET_GROUPS", "").strip()
        if seq_target_groups_raw:
            seq_target_groups = []
            for group_raw in seq_target_groups_raw.split(";"):
                group = [_strip_slash(x.strip()) for x in group_raw.split(",")
                         if x.strip()]
                if not group:
                    raise ValueError(
                        f"SEQUENTIAL_TARGET_GROUPS contains an empty group: {seq_target_groups_raw!r}"
                    )
                seq_target_groups.append(group)
        else:
            seq_target_groups = [[t] for t in seq_targets]

        seq_target_kv_port_groups_raw = os.getenv(
            "SEQUENTIAL_TARGET_KV_PORT_GROUPS", "").strip()
        if seq_target_kv_port_groups_raw:
            seq_target_kv_port_groups = []
            for group_raw in seq_target_kv_port_groups_raw.split(";"):
                group = [int(x.strip()) for x in group_raw.split(",")
                         if x.strip()]
                if not group:
                    raise ValueError(
                        "SEQUENTIAL_TARGET_KV_PORT_GROUPS contains an empty "
                        f"group: {seq_target_kv_port_groups_raw!r}"
                    )
                seq_target_kv_port_groups.append(group)
        else:
            seq_target_kv_port_groups = [[p] for p in kv_ports]

        if len(seq_target_groups) != len(seq_targets):
            raise ValueError(
                "SEQUENTIAL_TARGET_GROUPS size "
                f"({len(seq_target_groups)}) must equal SEQUENTIAL_TARGETS size "
                f"({len(seq_targets)})"
            )
        if len(seq_target_kv_port_groups) != len(seq_targets):
            raise ValueError(
                "SEQUENTIAL_TARGET_KV_PORT_GROUPS size "
                f"({len(seq_target_kv_port_groups)}) must equal "
                f"SEQUENTIAL_TARGETS size ({len(seq_targets)})"
            )
        for idx, group in enumerate(seq_target_groups):
            if len(group) != len(seq_target_kv_port_groups[idx]):
                raise ValueError(
                    "SEQUENTIAL_TARGET_GROUPS and "
                    "SEQUENTIAL_TARGET_KV_PORT_GROUPS must align per group: "
                    f"group_idx={idx} urls={group!r} kv_ports="
                    f"{seq_target_kv_port_groups[idx]!r}"
                )
            expected_dp = dp_sizes[idx]
            if expected_dp != len(group):
                raise ValueError(
                    "SEQUENTIAL_TARGET_DP_SIZES must equal target group size: "
                    f"group_idx={idx} dp_size={expected_dp} group={group!r}"
                )

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
            sequential_target_dp_sizes=dp_sizes,
            sequential_target_groups=seq_target_groups,
            sequential_target_kv_port_groups=seq_target_kv_port_groups,
            sequential_decode_tokens=seq_decode_tokens,
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "300")),
            connect_timeout_s=float(os.getenv("CONNECT_TIMEOUT_S", "30")),
            verbose_log=_parse_bool(os.getenv("PROXY_VERBOSE_LOG"), default=False),
            require_kv_transfer=_parse_bool(os.getenv("REQUIRE_KV_TRANSFER"), default=False),
            dynamic_kv_control_path=os.getenv("DYNAMIC_KV_CONTROL_PATH", "").strip(),
            dynamic_kv_wait_timeout_s=float(os.getenv("DYNAMIC_KV_WAIT_TIMEOUT_S", "120")),
            dynamic_kv_settle_s=float(os.getenv("DYNAMIC_KV_SETTLE_S", "2.0")),
            kv_owner_state_url=os.getenv("KV_OWNER_STATE_URL", "").strip(),
            kv_owner_state_strict=_parse_bool(os.getenv("KV_OWNER_STATE_STRICT"), default=False),
            kv_handoff_serial_barrier=_parse_bool(
                os.getenv("KV_HANDOFF_SERIAL_BARRIER"), default=True),
            kv_handoff_min_layers=max(
                1, int(os.getenv("KV_HANDOFF_MIN_LAYERS", "56"))),
            kv_handoff_wait_timeout_s=float(
                os.getenv("KV_HANDOFF_WAIT_TIMEOUT_S", "20")),
            kv_handoff_stable_polls=max(
                1, int(os.getenv("KV_HANDOFF_STABLE_POLLS", "2"))),
            kv_handoff_soft_min_layers=max(
                1, int(os.getenv("KV_HANDOFF_SOFT_MIN_LAYERS", "8"))),
            kv_handoff_global_phase_barrier=_parse_bool(
                os.getenv("KV_HANDOFF_GLOBAL_PHASE_BARRIER"), default=False),
            max_response_length=max(1, int(os.getenv("MAX_RESPONSE_LENGTH", "4096"))),
            upstream_max_model_len=max(
                1, int(os.getenv("UPSTREAM_MAX_MODEL_LEN", "3072"))),
        )


class RouteState:
    def __init__(self, cfg: ProxyConfig):
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._phase_cond = asyncio.Condition()
        self.decode_request_count = 0
        self.post_cutover_rr_index = 0
        self.decode_target_counts: dict[str, int] = {}
        self._phase = "hop1"
        self._phase_wave = 0
        self._released_wave = -1
        self._hop1_active = 0
        self._hop2_active = 0
        self._pending_hop2 = 0
        self._dp_rr_cursor_by_size: dict[int, int] = {}
        self._request_dp_ranks: dict[str, int] = {}
        self._active_dp_counts_by_size: dict[int, list[int]] = {}

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

    async def assign_dp_rank(
        self,
        request_id: str,
        dp_size: int,
        explicit_dp_rank: Optional[int] = None,
    ) -> int:
        if dp_size <= 1:
            return 0
        async with self._lock:
            existing = self._request_dp_ranks.get(request_id)
            if existing is not None:
                return existing
            if explicit_dp_rank is not None and explicit_dp_rank >= 0:
                rank = explicit_dp_rank % dp_size
            else:
                active = self._active_dp_counts_by_size.setdefault(
                    dp_size, [0] * dp_size)
                min_active = min(active)
                candidates = [
                    idx for idx, cnt in enumerate(active) if cnt == min_active
                ]
                cursor = self._dp_rr_cursor_by_size.get(dp_size, 0)
                rank = min(
                    candidates,
                    key=lambda idx: ((idx - cursor) % dp_size, idx),
                )
                self._dp_rr_cursor_by_size[dp_size] = (rank + 1) % dp_size
            self._request_dp_ranks[request_id] = rank
            active = self._active_dp_counts_by_size.setdefault(
                dp_size, [0] * dp_size)
            active[rank] += 1
            return rank

    async def release_dp_rank(self, request_id: str, dp_size: int) -> None:
        if dp_size <= 1:
            return
        async with self._lock:
            rank = self._request_dp_ranks.pop(request_id, None)
            if rank is None:
                return
            active = self._active_dp_counts_by_size.get(dp_size)
            if active is None or not (0 <= rank < len(active)):
                return
            active[rank] = max(0, active[rank] - 1)

    def _phase_barrier_enabled(self) -> bool:
        return (self.cfg.routing_mode == "sequential_handoff"
                and self.cfg.kv_handoff_global_phase_barrier
                and len(self.cfg.sequential_targets) == 2)

    async def phase_enter_hop1(self) -> int:
        if not self._phase_barrier_enabled():
            return 0
        async with self._phase_cond:
            while self._phase == "hop2":
                await self._phase_cond.wait()
            wave = self._phase_wave
            self._hop1_active += 1
            return wave

    async def phase_finish_hop1(self, wave: int, needs_hop2: bool) -> bool:
        if not self._phase_barrier_enabled():
            return needs_hop2
        async with self._phase_cond:
            self._hop1_active = max(0, self._hop1_active - 1)
            if needs_hop2:
                self._pending_hop2 += 1
            if self._hop1_active == 0:
                if self._pending_hop2 > 0:
                    self._phase = "hop2"
                    self._released_wave = wave
                else:
                    self._phase = "hop1"
                    self._phase_wave = max(self._phase_wave, wave + 1)
                self._phase_cond.notify_all()
            if not needs_hop2:
                return False
            while self._released_wave < wave:
                await self._phase_cond.wait()
            self._pending_hop2 = max(0, self._pending_hop2 - 1)
            self._hop2_active += 1
            return True

    async def phase_finish_hop2(self, wave: int) -> None:
        if not self._phase_barrier_enabled():
            return
        async with self._phase_cond:
            self._hop2_active = max(0, self._hop2_active - 1)
            if (self._phase == "hop2" and self._released_wave == wave
                    and self._hop2_active == 0 and self._pending_hop2 == 0):
                self._phase = "hop1"
                self._phase_wave = max(self._phase_wave, wave + 1)
                self._phase_cond.notify_all()

    async def phase_abort_hop1(self, wave: int) -> None:
        if not self._phase_barrier_enabled():
            return
        async with self._phase_cond:
            self._hop1_active = max(0, self._hop1_active - 1)
            if self._hop1_active == 0 and self._pending_hop2 == 0:
                self._phase = "hop1"
                self._phase_wave = max(self._phase_wave, wave + 1)
                self._phase_cond.notify_all()

    async def phase_abort_hop2(self, wave: int) -> None:
        if not self._phase_barrier_enabled():
            return
        async with self._phase_cond:
            self._hop2_active = max(0, self._hop2_active - 1)
            if (self._phase == "hop2" and self._released_wave == wave
                    and self._hop2_active == 0 and self._pending_hop2 == 0):
                self._phase = "hop1"
                self._phase_wave = max(self._phase_wave, wave + 1)
                self._phase_cond.notify_all()

def _resolve_stage_target(
    cfg: ProxyConfig,
    stage_idx: int,
    request_dp_rank_seed: Optional[int],
    base_request_id: str,
) -> tuple[str, int, int]:
    urls = cfg.sequential_target_groups[stage_idx]
    kv_ports = cfg.sequential_target_kv_port_groups[stage_idx]
    target_dp_size = len(urls)
    if request_dp_rank_seed is None:
        hop_dp_rank = 0
    else:
        hop_dp_rank = request_dp_rank_seed % max(1, target_dp_size)
    return urls[hop_dp_rank], kv_ports[hop_dp_rank], hop_dp_rank


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
                          prev_kv_port: int | None, prev_dp_rank: int | None,
                          next_target: str | None, next_kv_port: int | None,
                          next_dp_rank: int | None) -> str:
    rid = base_request_id
    if prev_target:
        phost, _ = _target_host_port(prev_target)
        rid += f"___prefill_addr_{phost}:{int(prev_kv_port)}"
        if prev_dp_rank is not None:
            rid += f"@dp{int(prev_dp_rank)}"
        rid += "___"
    if next_target:
        dhost, _ = _target_host_port(next_target)
        rid += f"___decode_addr_{dhost}:{int(next_kv_port)}"
        if next_dp_rank is not None:
            rid += f"@dp{int(next_dp_rank)}"
    return rid


def _merge_hop_text(existing: str, new: str, max_window: int = 4096) -> str:
    """Append hop text while removing longest suffix/prefix overlap."""
    if not existing:
        return new
    if not new:
        return existing
    # If upstream returns cumulative text for this request, keep the longest.
    if new.startswith(existing):
        return new
    if existing.startswith(new):
        return existing
    # If next hop restarts from an earlier prefix, trim the duplicated prefix
    # from new by finding the longest new-prefix already present in existing.
    max_k_anywhere = min(len(new), max_window)
    for k in range(max_k_anywhere, 31, -1):
        if new[:k] in existing:
            return existing + new[k:]
    max_k = min(len(existing), len(new), max_window)
    for k in range(max_k, 0, -1):
        if existing[-k:] == new[:k]:
            return existing + new[k:]
    return existing + new


def _is_cumulative_hop_text(existing: str, new: str) -> bool:
    if not existing or not new:
        return False
    if new.startswith(existing):
        return True
    # Heuristic: if new already contains the head of existing near start and
    # has comparable/greater length, it is likely cumulative text.
    head = existing[:min(64, len(existing))]
    pos = new.find(head)
    if pos != -1 and pos <= 96 and len(new) + 32 >= len(existing):
        return True
    return False


def _collapse_immediate_repeats(text: str,
                                min_chunk: int = 8,
                                max_chunk: int = 192) -> str:
    """Remove immediate duplicated chunks introduced at hop boundaries."""
    n = len(text)
    if n < min_chunk * 2:
        return text
    out: list[str] = []
    i = 0
    while i < n:
        matched = False
        max_l = min(max_chunk, (n - i) // 2)
        for l in range(max_l, min_chunk - 1, -1):
            a = text[i:i + l]
            b = text[i + l:i + 2 * l]
            if a == b:
                out.append(a)
                i += 2 * l
                while i + l <= n and text[i:i + l] == a:
                    i += l
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def _dedup_nearby_sentences(text: str, window: int = 4) -> str:
    """Drop repeated nearby sentences while preserving order."""
    parts = re.split(r"([.!?\n]+)", text)
    if len(parts) <= 2:
        return text
    out: list[str] = []
    recent: list[str] = []
    i = 0
    while i < len(parts):
        sent = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        norm = sent.strip().lower()
        if norm and len(norm) >= 12:
            if norm in recent[-window:]:
                i += 2
                continue
            recent.append(norm)
        out.append(sent)
        out.append(sep)
        i += 2
    return "".join(out)


def _write_dynamic_kv_control(path: str, active_upstream: str | None) -> None:
    if not path:
        return
    payload = {
        "active_upstream": active_upstream,
        "updated_at": time.time(),
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


async def _wait_upstream_ready(
    client: httpx.AsyncClient,
    upstream: str,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    url = upstream.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            resp = await client.get(url)
            if 200 <= resp.status_code < 300:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.25)
    return False


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
    async def kv_owner_acquire(req_id: str, worker: str, hop: int) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        try:
            resp = await client.post(
                cfg.kv_owner_state_url.rstrip("/") + "/acquire",
                json={"request_id": req_id, "worker": worker, "hop": hop},
            )
            return 200 <= resp.status_code < 300 and bool(resp.json().get("ok", False))
        except Exception:
            return False

    async def kv_owner_commit(req_id: str, worker: str, hop: int,
                              generated_tokens: int) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        try:
            resp = await client.post(
                cfg.kv_owner_state_url.rstrip("/") + "/commit",
                json={
                    "request_id": req_id,
                    "worker": worker,
                    "hop": hop,
                    "generated_tokens": generated_tokens,
                },
            )
            return 200 <= resp.status_code < 300 and bool(resp.json().get("ok", False))
        except Exception:
            return False

    async def kv_owner_release(req_id: str, worker: str, hop: int) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        try:
            resp = await client.post(
                cfg.kv_owner_state_url.rstrip("/") + "/release",
                json={"request_id": req_id, "worker": worker, "hop": hop},
            )
            return 200 <= resp.status_code < 300 and bool(resp.json().get("ok", False))
        except Exception:
            return False

    async def kv_owner_reset(req_id: str) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        try:
            resp = await client.post(
                cfg.kv_owner_state_url.rstrip("/") + "/reset",
                json={"request_id": req_id},
            )
            return 200 <= resp.status_code < 300 and bool(resp.json().get("ok", False))
        except Exception:
            return False

    async def kv_owner_wait_ready(req_id: str, min_layers: int, timeout_s: float,
                                  stable_polls: int, soft_min_layers: int,
                                  required_prev_hop: int) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        deadline = time.time() + max(0.1, timeout_s)
        good = 0
        while time.time() < deadline:
            try:
                resp = await client.get(
                    cfg.kv_owner_state_url.rstrip("/") + f"/state/{req_id}")
                if 200 <= resp.status_code < 300:
                    obj = resp.json()
                    if bool(obj.get("ok", False)):
                        n_layers = int(obj.get("num_kv_layers", 0))
                        committed_tokens = int(obj.get("committed_tokens", 0))
                        last_hop = int(obj.get("last_hop", 0))
                        # Primary readiness in serial-handoff mode:
                        # previous hop has committed generation for this request.
                        commit_ready = (
                            last_hop >= int(required_prev_hop)
                            and committed_tokens > 0
                        )
                        hard_ready = n_layers >= int(min_layers)
                        soft_ready = (
                            n_layers >= int(soft_min_layers)
                        )
                        if commit_ready or hard_ready or soft_ready:
                            good += 1
                            if good >= stable_polls:
                                return True
                        else:
                            good = 0
                    else:
                        good = 0
                else:
                    good = 0
            except Exception:
                good = 0
            await asyncio.sleep(0.1)
        return False

    async def kv_owner_wait_publish_done(req_id: str, required_hop: int,
                                         timeout_s: float) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        deadline = time.time() + max(0.1, timeout_s)
        logger.info("kv wait publish_done req=%s hop>=%s timeout=%.1fs",
                    req_id, required_hop, timeout_s)
        last_obj: dict[str, Any] | None = None
        while time.time() < deadline:
            try:
                resp = await client.get(
                    cfg.kv_owner_state_url.rstrip("/") + f"/state/{req_id}")
                if 200 <= resp.status_code < 300:
                    obj = resp.json()
                    if isinstance(obj, dict):
                        last_obj = obj
                    if (bool(obj.get("ok", False))
                            and int(obj.get("publish_done_hop", 0)) >=
                            int(required_hop)):
                        logger.info("kv publish_done satisfied req=%s state=%s",
                                    req_id, obj)
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.05)
        logger.error("kv publish_done timeout req=%s required_hop=%s state=%s",
                     req_id, required_hop, last_obj)
        return False

    async def kv_owner_wait_load_ack(req_id: str, required_hop: int,
                                     timeout_s: float) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        deadline = time.time() + max(0.1, timeout_s)
        logger.info("kv wait load_ack req=%s hop>=%s timeout=%.1fs",
                    req_id, required_hop, timeout_s)
        last_obj: dict[str, Any] | None = None
        while time.time() < deadline:
            try:
                resp = await client.get(
                    cfg.kv_owner_state_url.rstrip("/") + f"/state/{req_id}")
                if 200 <= resp.status_code < 300:
                    obj = resp.json()
                    if isinstance(obj, dict):
                        last_obj = obj
                    if (bool(obj.get("ok", False))
                            and int(obj.get("load_ack_hop", 0)) >=
                            int(required_hop)):
                        logger.info("kv load_ack satisfied req=%s state=%s",
                                    req_id, obj)
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.05)
        logger.error("kv load_ack timeout req=%s required_hop=%s state=%s",
                     req_id, required_hop, last_obj)
        return False

    async def kv_owner_resume(req_id: str, worker: str, hop: int) -> bool:
        if not cfg.kv_owner_state_url:
            return True
        try:
            resp = await client.post(
                cfg.kv_owner_state_url.rstrip("/") + "/resume",
                json={"request_id": req_id, "worker": worker, "hop": hop},
            )
            ok = 200 <= resp.status_code < 300 and bool(resp.json().get("ok", False))
            if ok:
                logger.info("kv resume posted req=%s worker=%s hop=%s",
                            req_id, worker, hop)
            else:
                logger.error("kv resume failed req=%s worker=%s hop=%s status=%s body=%s",
                             req_id, worker, hop, resp.status_code, resp.text)
            return ok
        except Exception:
            return False

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
    prompt_is_text = isinstance(prompt, str)
    prompt_is_token_ids = (
        isinstance(prompt, list)
        and all(isinstance(tok, int) for tok in prompt)
    )
    if not prompt_is_text and not prompt_is_token_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "sequential_handoff requires prompt as string or list[int]"},
        )

    total_max_tokens = int(req_obj.get("max_tokens", 16))
    if total_max_tokens > cfg.max_response_length:
        total_max_tokens = cfg.max_response_length
        req_obj["max_tokens"] = total_max_tokens
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
    cumulative_prompt_token_ids: Optional[list[int]] = None
    base_prompt = prompt
    req_kv = req_obj.get("kv_transfer_params")
    kv_transfer_params: Optional[dict[str, Any]]
    if isinstance(req_kv, dict):
        kv_transfer_params = dict(req_kv)
    else:
        # Default to local-only; enable remote transfer per-hop below.
        kv_transfer_params = {"do_remote_prefill": False, "do_remote_decode": False}

    sum_completion_tokens = 0
    all_completion_token_ids: list[int] = []
    all_token_logprobs: list[Any] = []
    first_prompt_token_ids: Optional[list[int]] = None
    first_prompt_tokens: Optional[int] = None
    last_resp: dict[str, Any] | None = None
    first_id = req_obj.get("request_id")
    base_request_id = str(first_id) if first_id else f"handoff-{uuid.uuid4().hex}"
    explicit_dp_rank = req_obj.get("data_parallel_rank")
    explicit_dp_rank = explicit_dp_rank if isinstance(explicit_dp_rank, int) and explicit_dp_rank >= 0 else None
    first_stage_dp_size = cfg.sequential_target_dp_sizes[0] if cfg.sequential_target_dp_sizes else 1
    request_dp_rank_seed = await state.assign_dp_rank(
        base_request_id,
        first_stage_dp_size,
        explicit_dp_rank=explicit_dp_rank,
    )
    # Prevent state bleed when clients accidentally reuse request_id.
    await kv_owner_reset(base_request_id)
    phase_wave = await state.phase_enter_hop1()
    hop1_phase_done = False
    hop2_phase_active = False
    try:
        for hop_idx, (target_base, hop_max_tokens) in enumerate(plan, start=1):
            is_last_hop = hop_idx == len(plan)
            stage_idx = hop_idx - 1
            target_dp_size = cfg.sequential_target_dp_sizes[stage_idx]
            hop_dp_rank = request_dp_rank_seed % max(1, target_dp_size)
            target_base = cfg.sequential_target_groups[stage_idx][hop_dp_rank]
            await state.add_target_hit(target_base)
            # Dynamic KV profile switch: only consumers (server2/3/4) are toggled.
            if cfg.dynamic_kv_control_path:
                # Only switch dynamic KV profile when the target is a consumer.
                # Avoid forcing "active->inactive" on every primary hop, which
                # causes unnecessary restart churn and can destabilize consumers.
                if target_base != cfg.primary_upstream:
                    _write_dynamic_kv_control(cfg.dynamic_kv_control_path, target_base)
                    if cfg.dynamic_kv_settle_s > 0:
                        await asyncio.sleep(cfg.dynamic_kv_settle_s)
                    ready = await _wait_upstream_ready(
                        client,
                        target_base,
                        cfg.dynamic_kv_wait_timeout_s,
                    )
                    if not ready:
                        return JSONResponse(
                            status_code=502,
                            content={
                                "error": "upstream_not_ready_after_dynamic_switch",
                                "upstream": target_base,
                                "hop": hop_idx,
                                "decode_idx": decode_idx,
                            },
                        )
            prev_target = None
            prev_kv_port = None
            next_target = None
            next_kv_port = None
            if hop_idx > 1:
                prev_stage_idx = stage_idx - 1
                prev_target = cfg.sequential_target_groups[prev_stage_idx][hop_dp_rank]
                prev_kv_port = cfg.sequential_target_kv_port_groups[prev_stage_idx][hop_dp_rank]
            if hop_idx < len(plan):
                next_stage_idx = stage_idx + 1
                next_target = cfg.sequential_target_groups[next_stage_idx][hop_dp_rank]
                next_kv_port = cfg.sequential_target_kv_port_groups[next_stage_idx][hop_dp_rank]

            hop_req = dict(req_obj)
            hop_req["stream"] = False
            if cfg.verbose_log:
                print(
                    "[proxy:sequential_handoff] "
                    f"base_request_id={base_request_id} hop={hop_idx}/{len(plan)} "
                    f"target={target_base} target_dp_size={target_dp_size} "
                    f"explicit_dp_rank={explicit_dp_rank} chosen_dp_rank={hop_dp_rank}"
                )
            # For non-first hops, include text generated so far in prompt so the
            # destination scheduler can account for full context length. KV handoff
            # remains enabled and should avoid recomputing most of this context.
            if hop_idx == 1:
                hop_req["prompt"] = base_prompt
            elif cumulative_prompt_token_ids:
                # Use exact token history from previous hop to avoid text->token
                # re-encoding drift across servers.
                hop_req["prompt"] = cumulative_prompt_token_ids
            else:
                # Fallback path when upstream did not return token IDs.
                if isinstance(base_prompt, str):
                    hop_req["prompt"] = base_prompt + generated_text
                else:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "missing_token_history_for_token_prompt",
                            "detail": (
                                "Upstream did not return cumulative token ids, "
                                "cannot continue handoff from token-id prompt."
                            ),
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
                    )
            # Guard against per-hop context overflow:
            # input_tokens + max_tokens must not exceed upstream max_model_len.
            effective_hop_max_tokens = int(hop_max_tokens)
            prompt_obj = hop_req.get("prompt")
            if isinstance(prompt_obj, list):
                prompt_len = len(prompt_obj)
                remain_budget = int(cfg.upstream_max_model_len) - int(prompt_len)
                if remain_budget <= 0:
                    if last_resp is not None:
                        # No decode budget left at this hop. End chain gracefully.
                        choices_obj = last_resp.get("choices")
                        if (isinstance(choices_obj, list) and choices_obj
                                and isinstance(choices_obj[0], dict)):
                            choices_obj[0]["finish_reason"] = "length"
                        break
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "context_overflow_before_hop",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "prompt_tokens": prompt_len,
                            "max_model_len": int(cfg.upstream_max_model_len),
                            "decode_idx": decode_idx,
                        },
                    )
                effective_hop_max_tokens = min(
                    effective_hop_max_tokens, int(remain_budget))
            hop_req["max_tokens"] = max(1, int(effective_hop_max_tokens))
            # Internal chaining requires exact token continuity across hops.
            hop_req["return_token_ids"] = True
            # Keep a stable logical request id and embed prefill/decode addrs for
            # connectors (notably P2pNcclConnector) to resolve recv/send peers.
            hop_req["request_id"] = _build_p2p_request_id(
                base_request_id,
                prev_target,
                prev_kv_port,
                hop_dp_rank if hop_idx > 1 else None,
                next_target,
                next_kv_port,
                hop_dp_rank if hop_idx < len(plan) else None,
            )
            if (cfg.sequential_target_dp_sizes[stage_idx] > 1
                    and len(cfg.sequential_target_groups[stage_idx]) > 1):
                hop_req.pop("data_parallel_rank", None)
            else:
                hop_req["data_parallel_rank"] = hop_dp_rank
            fast_phase_handoff = (
                cfg.kv_handoff_global_phase_barrier
                and len(cfg.sequential_targets) == 2
                and hop_idx > 1
            )
            if hop_idx > 1 and not fast_phase_handoff:
                if not await kv_owner_wait_publish_done(
                        base_request_id, hop_idx - 1,
                        cfg.kv_handoff_wait_timeout_s):
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_publish_done_timeout",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
                    )
            if hop_idx > 1 and cfg.kv_handoff_serial_barrier and not fast_phase_handoff:
                ready = await kv_owner_wait_ready(
                    base_request_id,
                    cfg.kv_handoff_min_layers,
                    cfg.kv_handoff_wait_timeout_s,
                    cfg.kv_handoff_stable_polls,
                    cfg.kv_handoff_soft_min_layers,
                    hop_idx - 1,
                )
                if not ready:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_handoff_barrier_timeout",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                            "min_layers": cfg.kv_handoff_min_layers,
                            "soft_min_layers": cfg.kv_handoff_soft_min_layers,
                        },
                    )
            if not await kv_owner_acquire(base_request_id, target_base, hop_idx):
                if cfg.kv_owner_state_strict:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_owner_acquire_failed",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
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
            req_task = asyncio.create_task(
                client.request(
                    method="POST",
                    url=upstream_url,
                    headers=req_headers,
                    content=json.dumps(hop_req).encode("utf-8"),
                ))
            if hop_idx > 1 and not fast_phase_handoff:
                acked = await kv_owner_wait_load_ack(
                    base_request_id, hop_idx - 1,
                    cfg.kv_handoff_wait_timeout_s)
                if not acked:
                    req_task.cancel()
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_load_ack_timeout",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
                    )
                if not await kv_owner_resume(base_request_id, target_base,
                                             hop_idx - 1):
                    req_task.cancel()
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_resume_failed",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
                    )
            try:
                resp = await req_task
            except httpx.HTTPError as exc:
                if hop_idx == 1 and not hop1_phase_done:
                    await state.phase_abort_hop1(phase_wave)
                elif hop_idx > 1 and hop2_phase_active:
                    await state.phase_abort_hop2(phase_wave)
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
            except asyncio.CancelledError:
                if hop_idx == 1 and not hop1_phase_done:
                    await state.phase_abort_hop1(phase_wave)
                elif hop_idx > 1 and hop2_phase_active:
                    await state.phase_abort_hop2(phase_wave)
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": "cancelled_handoff_request",
                        "upstream": target_base,
                        "path": full_path,
                        "decode_idx": decode_idx,
                        "hop": hop_idx,
                    },
                )

            if resp.status_code >= 400:
                if hop_idx == 1 and not hop1_phase_done:
                    await state.phase_abort_hop1(phase_wave)
                elif hop_idx > 1 and hop2_phase_active:
                    await state.phase_abort_hop2(phase_wave)
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
                if hop_idx == 1 and not hop1_phase_done:
                    await state.phase_abort_hop1(phase_wave)
                elif hop_idx > 1 and hop2_phase_active:
                    await state.phase_abort_hop2(phase_wave)
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
            # Hop outputs are not always strictly incremental across engines.
            # Some paths may return cumulative text; merge defensively.
            if not generated_text:
                generated_text = hop_text
            elif _is_cumulative_hop_text(generated_text, hop_text):
                generated_text = _merge_hop_text(generated_text, hop_text)
            else:
                generated_text = generated_text + hop_text
            choices_obj = resp_obj.get("choices")
            if isinstance(choices_obj, list) and choices_obj:
                ch0 = choices_obj[0] if isinstance(choices_obj[0], dict) else {}
                p_ids = ch0.get("prompt_token_ids")
                o_ids = ch0.get("token_ids")
                logprobs_obj = ch0.get("logprobs") if isinstance(ch0, dict) else None
                if isinstance(o_ids, list) and all(isinstance(x, int) for x in o_ids):
                    all_completion_token_ids.extend(int(x) for x in o_ids)
                    if isinstance(p_ids, list) and all(isinstance(x, int) for x in p_ids):
                        if first_prompt_token_ids is None:
                            first_prompt_token_ids = list(p_ids)
                        cumulative_prompt_token_ids = list(p_ids) + list(o_ids)
                    elif cumulative_prompt_token_ids is not None:
                        cumulative_prompt_token_ids = cumulative_prompt_token_ids + list(o_ids)
                if isinstance(logprobs_obj, dict):
                    token_logprobs = logprobs_obj.get("token_logprobs")
                    if isinstance(token_logprobs, list):
                        all_token_logprobs.extend(token_logprobs)
            usage_obj = resp_obj.get("usage") or {}
            hop_completion_tokens = (
                int(usage_obj.get("completion_tokens", 0))
                if isinstance(usage_obj.get("completion_tokens"), int) else 0
            )
            if not await kv_owner_commit(
                    base_request_id, target_base, hop_idx,
                    hop_completion_tokens):
                if cfg.kv_owner_state_strict:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": "kv_owner_commit_failed",
                            "upstream": target_base,
                            "hop": hop_idx,
                            "decode_idx": decode_idx,
                        },
                    )
            next_kv_transfer_params = resp_obj.get("kv_transfer_params")
            if (not is_last_hop and cfg.require_kv_transfer
                    and not isinstance(next_kv_transfer_params, dict)):
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

            # Preserve native vLLM termination semantics: if this hop already
            # reached a terminal finish_reason (not token-limit), stop handoff.
            finish_reason = None
            choices_obj = resp_obj.get("choices")
            if (isinstance(choices_obj, list) and choices_obj
                    and isinstance(choices_obj[0], dict)):
                finish_reason = choices_obj[0].get("finish_reason")
            should_continue = (finish_reason is None or str(finish_reason) == "length")
            if hop_idx == 1 and not hop1_phase_done:
                hop1_phase_done = True
                hop2_phase_active = await state.phase_finish_hop1(
                    phase_wave, needs_hop2=(not is_last_hop and should_continue))
            if finish_reason is not None and str(finish_reason) != "length":
                break

        assert last_resp is not None
        if hop2_phase_active:
            await state.phase_finish_hop2(phase_wave)
        if len(plan) > 1:
            await kv_owner_release(base_request_id, plan[-1][0], len(plan))
        if "choices" in last_resp and isinstance(last_resp["choices"], list) and last_resp["choices"]:
            last_resp["choices"][0]["text"] = generated_text
            if all_completion_token_ids:
                last_resp["choices"][0]["token_ids"] = all_completion_token_ids
            if first_prompt_token_ids is not None:
                last_resp["choices"][0]["prompt_token_ids"] = first_prompt_token_ids
            logprobs_obj = last_resp["choices"][0].get("logprobs")
            if isinstance(logprobs_obj, dict) and all_token_logprobs:
                logprobs_obj["token_logprobs"] = all_token_logprobs
        usage = last_resp.get("usage")
        if isinstance(usage, dict):
            if first_prompt_tokens is not None:
                usage["prompt_tokens"] = first_prompt_tokens
            usage["completion_tokens"] = sum_completion_tokens
            if first_prompt_tokens is not None:
                usage["total_tokens"] = first_prompt_tokens + sum_completion_tokens
        last_resp["kv_transfer_params"] = kv_transfer_params
        return JSONResponse(status_code=200, content=last_resp)
    finally:
        await state.release_dp_rank(base_request_id, first_stage_dp_size)


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
            "sequential_target_dp_sizes": cfg.sequential_target_dp_sizes,
            "sequential_target_groups": cfg.sequential_target_groups,
            "sequential_target_kv_port_groups": cfg.sequential_target_kv_port_groups,
            "sequential_decode_tokens": cfg.sequential_decode_tokens,
            "kv_handoff_serial_barrier": cfg.kv_handoff_serial_barrier,
            "kv_handoff_min_layers": cfg.kv_handoff_min_layers,
            "kv_handoff_wait_timeout_s": cfg.kv_handoff_wait_timeout_s,
            "kv_handoff_soft_min_layers": cfg.kv_handoff_soft_min_layers,
            "kv_handoff_global_phase_barrier": cfg.kv_handoff_global_phase_barrier,
            "decode_request_count": state.decode_request_count,
            "decode_target_counts": state.decode_target_counts,
            "post_cutover_rr_index": state.post_cutover_rr_index,
            "max_response_length": cfg.max_response_length,
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
