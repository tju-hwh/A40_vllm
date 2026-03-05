from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class RequestState:
    request_id: str
    owner: Optional[str] = None
    epoch: int = 0
    committed_tokens: int = 0
    last_hop: int = 0
    updated_at: float = field(default_factory=time.time)
    # layer_name -> KV export metadata
    kv_by_layer: dict[str, dict[str, Any]] = field(default_factory=dict)


class AcquireReq(BaseModel):
    request_id: str
    worker: str
    hop: int


class CommitReq(BaseModel):
    request_id: str
    worker: str
    hop: int
    generated_tokens: int = 0


class ReleaseReq(BaseModel):
    request_id: str
    worker: str
    hop: int


class RegisterKVReq(BaseModel):
    request_id: str
    worker: str
    hop: int
    layer_name: str
    tensor_key: str
    num_tokens: int = 0
    num_blocks: int = 0
    block_ids: list[int] = []


class LookupKVReq(BaseModel):
    request_id: str
    layer_name: str
    min_tokens: int = 0
    min_blocks: int = 0


class ResetReq(BaseModel):
    request_id: str


def create_app() -> FastAPI:
    app = FastAPI(title="KV Owner State Server", version="0.1.0")
    states: dict[str, RequestState] = {}

    def _canonical_request_id(request_id: str) -> str:
        # Router control-plane may use user request_id (e.g. stage2-xxx),
        # while engine/KV data-plane may prepend completion prefixes
        # (e.g. cmpl-stage2-xxx). Keep one canonical key for both.
        if request_id.startswith("cmpl-"):
            return request_id[len("cmpl-"):]
        return request_id

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/acquire")
    async def acquire(req: AcquireReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            st = RequestState(request_id=rid)
            states[rid] = st
        st.owner = req.worker
        st.epoch += 1
        st.last_hop = req.hop
        st.updated_at = time.time()
        return {
            "ok": True,
            "request_id": st.request_id,
            "owner": st.owner,
            "epoch": st.epoch,
            "committed_tokens": st.committed_tokens,
            "last_hop": st.last_hop,
        }

    @app.post("/commit")
    async def commit(req: CommitReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            return {"ok": False, "error": "unknown_request"}
        if st.owner != req.worker:
            return {"ok": False, "error": "owner_mismatch", "owner": st.owner}
        st.committed_tokens += int(req.generated_tokens)
        st.last_hop = req.hop
        st.updated_at = time.time()
        return {
            "ok": True,
            "request_id": st.request_id,
            "owner": st.owner,
            "epoch": st.epoch,
            "committed_tokens": st.committed_tokens,
            "last_hop": st.last_hop,
        }

    @app.post("/release")
    async def release(req: ReleaseReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            return {"ok": False, "error": "unknown_request"}
        if st.owner == req.worker:
            st.owner = None
            st.last_hop = req.hop
            st.updated_at = time.time()
        return {
            "ok": True,
            "request_id": st.request_id,
            "owner": st.owner,
            "epoch": st.epoch,
            "committed_tokens": st.committed_tokens,
            "last_hop": st.last_hop,
        }

    @app.post("/register_kv")
    async def register_kv(req: RegisterKVReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            st = RequestState(request_id=rid)
            states[rid] = st
        st.kv_by_layer[req.layer_name] = {
            "tensor_key": req.tensor_key,
            "num_tokens": int(req.num_tokens),
            "num_blocks": int(req.num_blocks),
            "block_ids": [int(x) for x in req.block_ids],
            "worker": req.worker,
            "hop": int(req.hop),
            "updated_at": time.time(),
        }
        st.updated_at = time.time()
        return {"ok": True}

    @app.post("/lookup_kv")
    async def lookup_kv(req: LookupKVReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            return {"ok": False, "error": "unknown_request"}
        rec = st.kv_by_layer.get(req.layer_name)
        if rec is None:
            return {"ok": False, "error": "unknown_layer"}
        if int(rec.get("num_tokens", 0)) < int(req.min_tokens):
            return {"ok": False, "error": "insufficient_tokens", "record": rec}
        if int(rec.get("num_blocks", 0)) < int(req.min_blocks):
            return {"ok": False, "error": "insufficient_blocks", "record": rec}
        return {"ok": True, "record": rec}

    @app.get("/state/{request_id}")
    async def get_state(request_id: str) -> dict[str, Any]:
        st = states.get(_canonical_request_id(request_id))
        if st is None:
            return {"ok": False, "error": "unknown_request"}
        return {
            "ok": True,
            "request_id": st.request_id,
            "owner": st.owner,
            "epoch": st.epoch,
            "committed_tokens": st.committed_tokens,
            "last_hop": st.last_hop,
            "num_kv_layers": len(st.kv_by_layer),
            "updated_at": st.updated_at,
        }

    @app.post("/reset")
    async def reset(req: ResetReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        states.pop(rid, None)
        return {"ok": True, "request_id": rid}

    return app
