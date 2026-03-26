from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class RequestState:
    request_id: str
    owner: Optional[str] = None
    epoch: int = 0
    committed_tokens: int = 0
    last_hop: int = 0
    updated_at: float = field(default_factory=time.time)
    publish_done_hop: int = 0
    load_ack_hop: int = 0
    resume_hop: int = 0
    acquired_at: Optional[float] = None
    commit_updated_at: Optional[float] = None
    publish_done_at: dict[int, float] = field(default_factory=dict)
    load_ack_at: dict[int, float] = field(default_factory=dict)
    resume_at: dict[int, float] = field(default_factory=dict)
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


class RegisterKVBatchReq(BaseModel):
    items: list[RegisterKVReq]


class PublishDoneReq(BaseModel):
    request_id: str
    worker: str
    hop: int


class LoadAckReq(BaseModel):
    request_id: str
    worker: str
    hop: int


class ResumeReq(BaseModel):
    request_id: str
    worker: str
    hop: int


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
        now = time.time()
        st.acquired_at = now if st.acquired_at is None else st.acquired_at
        st.updated_at = now
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
        now = time.time()
        st.commit_updated_at = now
        st.updated_at = now
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
        new_num_tokens = int(req.num_tokens)
        new_num_blocks = int(req.num_blocks)
        new_block_ids = [int(x) for x in req.block_ids]
        old = st.kv_by_layer.get(req.layer_name)
        if old is not None:
            old_num_tokens = int(old.get("num_tokens", 0))
            old_num_blocks = int(old.get("num_blocks", 0))
            old_block_ids = [int(x) for x in old.get("block_ids", [])]
            # KV metadata should be monotonic for a request/layer. Never let a
            # shorter update overwrite a longer one, otherwise downstream
            # lookup may observe regressed block coverage and produce dst/src
            # mismatches during handoff.
            if old_num_tokens > new_num_tokens:
                new_num_tokens = old_num_tokens
            if old_num_blocks > new_num_blocks:
                new_num_blocks = old_num_blocks
                new_block_ids = old_block_ids
            elif len(old_block_ids) > len(new_block_ids):
                # Keep the longer block-id list if block counts tie or caller
                # reports fewer ids than previously registered.
                new_block_ids = old_block_ids
        st.kv_by_layer[req.layer_name] = {
            "tensor_key": req.tensor_key,
            "num_tokens": new_num_tokens,
            "num_blocks": new_num_blocks,
            "block_ids": new_block_ids,
            "worker": req.worker,
            "hop": int(req.hop),
            "updated_at": time.time(),
        }
        st.updated_at = time.time()
        return {"ok": True}

    @app.post("/register_kv_batch")
    async def register_kv_batch(req: RegisterKVBatchReq) -> dict[str, Any]:
        for item in req.items:
            await register_kv(item)
        return {"ok": True, "count": len(req.items)}

    @app.post("/publish_done")
    async def publish_done(req: PublishDoneReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            st = RequestState(request_id=rid)
            states[rid] = st
        st.publish_done_hop = max(int(st.publish_done_hop), int(req.hop))
        now = time.time()
        st.publish_done_at[int(req.hop)] = now
        st.updated_at = now
        logger.info("publish_done req=%s worker=%s hop=%s", rid, req.worker,
                    req.hop)
        return {"ok": True, "publish_done_hop": st.publish_done_hop}

    @app.post("/load_ack")
    async def load_ack(req: LoadAckReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            st = RequestState(request_id=rid)
            states[rid] = st
        st.load_ack_hop = max(int(st.load_ack_hop), int(req.hop))
        st.resume_hop = max(int(st.resume_hop), int(req.hop))
        now = time.time()
        st.load_ack_at[int(req.hop)] = now
        st.resume_at[int(req.hop)] = now
        st.updated_at = now
        logger.info("load_ack req=%s worker=%s hop=%s", rid, req.worker,
                    req.hop)
        return {
            "ok": True,
            "load_ack_hop": st.load_ack_hop,
            "resume_hop": st.resume_hop,
        }

    @app.post("/resume")
    async def resume(req: ResumeReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        st = states.get(rid)
        if st is None:
            st = RequestState(request_id=rid)
            states[rid] = st
        st.resume_hop = max(int(st.resume_hop), int(req.hop))
        now = time.time()
        st.resume_at[int(req.hop)] = now
        st.updated_at = now
        logger.info("resume req=%s worker=%s hop=%s", rid, req.worker,
                    req.hop)
        return {"ok": True, "resume_hop": st.resume_hop}

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
            "publish_done_hop": st.publish_done_hop,
            "load_ack_hop": st.load_ack_hop,
            "resume_hop": st.resume_hop,
            "num_kv_layers": len(st.kv_by_layer),
            "acquired_at": st.acquired_at,
            "commit_updated_at": st.commit_updated_at,
            "publish_done_at": st.publish_done_at,
            "load_ack_at": st.load_ack_at,
            "resume_at": st.resume_at,
            "updated_at": st.updated_at,
        }

    @app.post("/reset")
    async def reset(req: ResetReq) -> dict[str, Any]:
        rid = _canonical_request_id(req.request_id)
        states.pop(rid, None)
        return {"ok": True, "request_id": rid}

    @app.post("/reset_all")
    async def reset_all() -> dict[str, Any]:
        count = len(states)
        states.clear()
        logger.info("reset_all cleared owner-state records=%s", count)
        return {"ok": True, "cleared": count}

    return app
