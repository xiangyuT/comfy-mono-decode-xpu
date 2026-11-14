# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MTP decode with the per-step bookkeeping on the device and no per-step host sync.

Upstream's `_generate_mtp` ends every step with one sync (the sampled tokens), walks the accepted path
on the host, and prepares the next step from host counters (FixedKVState / LinearKV prepare, rollback,
commit_path, the deferred-delta ctl row, nt_buf / h_buf, pos). The next step's first draft therefore
starts on an idle device after the host has done all of that. Here the walk, the counters, the ctl row,
the KV slot moves and the next token / hidden row are computed by small device kernels
(csrc/mtp_loop.sycl), and the host reads each step's committed tokens one step late from a pinned ring,
after it has already queued the next step. Drafts, the verify forward, lm_head and sampling are
upstream's calls (the experimental MONO_DECODE_XPU_DRAFT_INT4 replaces only the drafts' lm_head pick:
draft_lmhead.py), so with a fixed tree the tokens are the ones upstream's loop would produce from the
same values. With `auto`, the asynchronous loop applies a tree change one step later than upstream
(it reads each step one step late); "sync" and "shadow" apply it at upstream's step.

Modes (MONO_DECODE_XPU_DEVICE_MTP): "1" asynchronous (default), "sync" one sync per step, "shadow" one
sync per step plus upstream's host walk on the same values, compared field by field. Anything outside
the supported shape (non-XPU, batch > 1, penalties, other cache types) runs upstream's function.
The asynchronous loop queues one step past a stop token or max_length; its tokens are dropped.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

import torch

from . import autotree, draft_head, draft_lmhead, guard, tiles, verify_head
from . import sampling as _sampling  # (_device_generate's `sampling` is the sampling parameters)

CLAIM = "__comfy_mono_decode_xpu_mtp_loop__"
MODE = os.environ.get("MONO_DECODE_XPU_DEVICE_MTP", "1")
# the draft head's XPU graph saves host time, which the asynchronous loop already hides (measured: no gain)
GRAPH_IN_LOOP = os.environ.get("MONO_DECODE_XPU_DEVICE_MTP_GRAPH", "0") == "1"
_tls = threading.local()


def in_async_loop():
    """True while this thread runs the asynchronous device loop (graph_draft stays eager there)."""
    return getattr(_tls, "async_loop", False) and not GRAPH_IN_LOOP


def in_device_loop():
    """True while this thread runs the device loop in any mode (it times the auto tree itself)."""
    return getattr(_tls, "device_loop", False)


# Upstream code the device loop reimplements (ComfyUI 0.45): _generate_mtp, the KV classes whose host
# state it keeps on the device, the verify tree and MTP_MODES. Different source (another ComfyUI
# version, or another node's wrapper of _generate_mtp) runs upstream's loop.
_UPSTREAM = "c980db6478a362ec"


def _fingerprint(llama, q35, original):
    import hashlib
    import inspect
    h = hashlib.sha256()
    for obj in (original, llama.FixedKV, llama.FixedKVState, llama.FixedKVCache, q35.LinearKV, q35.VerifyTree,
                q35.tree_spec):
        h.update(inspect.getsource(obj).encode())
    h.update(repr(q35.MTP_MODES).encode())
    return h.hexdigest()[:16]
RING = 8
_lock = threading.Lock()
_stats = {"generates": 0, "steps": 0, "fallback": {}, "shadow_mismatch": 0, "shadow_checked": 0, "host_rows": 0,
          "draft_int4": 0, "draft_int4_checked": 0, "draft_int4_mismatch": 0, "draft_int4_generates": 0}


def stats():
    with _lock:
        return dict(_stats, fallback=dict(_stats["fallback"]), draft_head=draft_head.stats(), verify_head=verify_head.stats())


def _fallback(why):
    with _lock:
        _stats["fallback"][why] = _stats["fallback"].get(why, 0) + 1
    return None


def _tree_tensors(tree, device):
    """children [8][8] (-1 padded) and the ctl program tail (parent[8], prog[8]) of a VerifyTree."""
    from comfy.text_encoders.llama import SPEC_ROWS
    children = [[-1] * SPEC_ROWS for _ in range(SPEC_ROWS)]
    for r, cs in enumerate(tree.children[:SPEC_ROWS]):
        for j, c in enumerate(cs[:SPEC_ROWS]):
            children[r][j] = c
    ch = torch.tensor(children, dtype=torch.int32).reshape(-1)
    ctl = tree.ctl.to(torch.int32).reshape(-1)
    return ch.to(device), ctl.to(device)


def _install_device_kv(llama, q35):
    """prepare (FixedKVState, LinearKV) and advance (FixedKV, their base) become no-ops for caches a device
    loop owns: their position / seqlen / ctl and the committed length live on the device."""
    done = 0
    for cls, names in ((llama.FixedKV, ("advance",)), (llama.FixedKVState, ("prepare",)), (q35.LinearKV, ("prepare",))):
        for name in names:
            original = cls.__dict__.get(name)
            if original is None or getattr(original, CLAIM, False):
                continue

            def make(original=original):
                def method(self, num_tokens):
                    if getattr(self, "_mono_device", False):
                        return None
                    return original(self, num_tokens)
                setattr(method, CLAIM, True)
                method.__wrapped__ = original
                return method
            setattr(cls, name, make())
            done += 1
    return done


def _upstream_walk(children, toks, dr, depth, cand):
    """Upstream's host accept walk (qwen35 _generate_mtp step) on host lists, for shadow checks."""
    path = [0]
    while True:
        node = next((c for c in children[path[-1]] if dr[c] == toks[path[-1]]), None)
        if node is None:
            break
        path.append(node)
    ranks = []
    if cand is not None:
        for node in path:
            if node >= depth:
                break
            candidates = cand[7 * node:7 * node + 7]
            ranks.append(next((r for r, token in enumerate(candidates) if token == toks[node]), 7))
    commit = [dr[r] for r in path[1:]] + [toks[path[-1]]]
    row = path[-1]
    on_chain = row == len(path) - 1
    kept = min(len(path) if on_chain else len(path) - 1, depth)
    return path, commit, ranks, row, on_chain, kept


def install():
    from comfy.text_encoders import llama
    from comfy.text_encoders import qwen35 as q35
    cls = getattr(q35, "Qwen35", None)
    original = getattr(cls, "_generate_mtp", None)
    if original is None or getattr(original, CLAIM, False) or MODE == "0":
        return False
    for name in ("VerifyTree", "MTP_MODES", "tree_spec", "best_widths", "LinearKV"):
        if not hasattr(q35, name):
            return False
    try:
        fingerprint = _fingerprint(llama, q35, original)
    except (OSError, TypeError):
        fingerprint = None
    if fingerprint != _UPSTREAM:
        _fallback("upstream_changed")
        return False
    _install_device_kv(llama, q35)

    def _generate_mtp(self, embeds, max_length, stop_tokens, mtp, sampling=None, position_ids=None,
                      repetition_penalty=1.0, presence_penalty=0.0):
        call = lambda: original(self, embeds, max_length, stop_tokens, mtp, sampling, position_ids,
                                repetition_penalty, presence_penalty)
        device = getattr(embeds, "device", None)
        if (guard.off("device_mtp") or device is None or device.type != "xpu" or getattr(self, "mtp", None) is None
                or not tiles.session_active() or llama.penalty_active(repetition_penalty, presence_penalty)
                or (embeds.dim() == 3 and embeds.shape[0] != 1) or not mtp):
            return call()
        try:
            from . import _C
            for fn in ("mtp_prep_draft", "mtp_prep_verify", "mtp_accept"):
                if not hasattr(_C, fn):
                    return call()
        except Exception:
            return call()
        try:
            out = _device_generate(self, _C, llama, q35, embeds, max_length, stop_tokens, mtp, sampling, position_ids)
        except Exception as e:  # re-raises a user interrupt; otherwise switched off, upstream reruns the generate
            guard.failed("device_mtp", e)  # (its caches are created per call: nothing of this one is reused)
            return call()
        return call() if out is None else out

    setattr(_generate_mtp, CLAIM, True)
    _generate_mtp.__wrapped__ = original
    cls._generate_mtp = _generate_mtp
    return True


def _device_generate(self, _C, llama, q35, embeds, max_length, stop_tokens, mtp, sampling, position_ids):
    """Upstream's _generate_mtp (ComfyUI 0.45) with the decode loop's bookkeeping on the device.
    Returns None before the prefill when the caches are not of the supported kinds."""
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.ops
    import comfy.utils
    import torch.nn.functional as F
    from tqdm import tqdm
    from comfy.text_encoders.llama import SPEC_ROWS, rope_matrix

    device = embeds.device
    cfg = self.model.config
    if stop_tokens is None:
        stop_tokens = cfg.stop_tokens
    dt = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    embeds = embeds.to(dt)
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    # upstream's capacity plus one more verify step: the asynchronous loop may queue one step past the end
    cap = embeds.shape[1] + max_length + 7 + SPEC_ROWS
    pkv = self.init_kv_cache(embeds.shape[0], cap, device, dt)
    mtp_kv = self.mtp.layers[0].self_attn.init_kv_cache(embeds.shape[0], cap, device, dt, {})
    attn = [kv for kv in pkv if not isinstance(kv, q35.LinearKV)]
    lin = [kv for kv in pkv if isinstance(kv, q35.LinearKV)]
    if (dt != torch.bfloat16 or not attn or not lin
            or not all(type(kv) is llama.FixedKVCache and kv.key.dtype == torch.bfloat16 and kv.key.is_contiguous()
                       and kv.value.is_contiguous() and kv.position is attn[0].position and kv.seqlen is attn[0].seqlen
                       for kv in attn)
            or not all(kv.ctl is lin[0].ctl for kv in lin) or type(mtp_kv) is not llama.FixedKVCache
            or lin[0].ctl.numel() != 2 + 3 * SPEC_ROWS):
        return _fallback("caches")
    head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens

    def verify_logits(x):
        if vh is not None:
            return vh.logits(x)
        if not head.comfy_cast_weights:
            return F.linear(x, head.weight.to(x), None)
        with comfy.ops.CastBiasWeightContext(head, x, offloadable=True) as (w, _bias):
            return F.linear(x, w)

    generator = None
    if sampling is not None:
        generator = torch.Generator(device=device).manual_seed(sampling["seed"])
    rope_offset = 0 if position_ids is None else int(position_ids[:, -1].max()) + 1 - embeds.shape[1]
    ftab = rope_matrix(self.model.compute_freqs_cis(
        torch.arange(rope_offset, cap + rope_offset, device=device, dtype=torch.float).unsqueeze(0), device))
    nt_buf = torch.empty((embeds.shape[0], 1), device=device, dtype=torch.long)
    h_buf = torch.empty((embeds.shape[0], 1, cfg.hidden_size), device=device, dtype=dt)
    x, _, _ = self.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=pkv, position_ids=position_ids)
    lg0 = self.logits(x)[:, -1]
    if sampling is None:
        nt_buf.copy_(lg0.argmax(dim=-1, keepdim=True))
    else:
        nt_buf.copy_(self.sample_token(lg0, sampling["temperature"], sampling["top_k"], sampling["top_p"],
                                       sampling["min_p"], 1.0, [], generator))
    h_buf.copy_(x[:, -1:, :])
    del x, lg0
    if hasattr(self.model.embed_tokens, "_v"):
        vbar = self.model.embed_tokens._v[0]
        comfy.model_management.reset_cast_buffers()
        vbar.set_watermark(vbar.max_size)
    ids = [nt_buf[0].item()]
    pos = embeds.shape[1]
    progress = comfy.utils.ProgressBar(max_length)
    console = tqdm(total=max_length, desc="Generating tokens", initial=1)

    def update_progress(n):
        progress.update(n)
        console.update(n)

    # experimental two-stage draft lm_head (MONO_DECODE_XPU_DRAFT_INT4): None unless enabled and eligible
    lm2 = draft_lmhead.DraftLMHead.create(_C, head, device) if draft_lmhead.enabled() else None
    lm2_on = lm2 is not None and draft_lmhead.MODE == "1"
    # the fused draft head (MONO_DECODE_XPU_DRAFT_FUSED): None runs the eager head
    dh = draft_head.DraftHead.create(_C, self, device, lm2 if lm2_on else None) if draft_head.enabled() else None
    vh = verify_head.VerifyHead.create(_C, self, device)
    adaptive = mtp is True
    widths = (4, 3)
    hits = [[0.0] * 8 for _ in range(7)]
    steps, draft_ms, verify_ms = 0, 0.0, 0.0
    draft_ranks = torch.zeros((7, 7), device=device, dtype=torch.long) if adaptive else None
    capacity, spec = q35.MTP_MODES["auto" if adaptive else mtp]
    tree = q35.VerifyTree(q35.tree_spec(widths) if adaptive else spec, capacity, device)
    depth = tree.chain
    for kv in pkv:
        if isinstance(kv, q35.LinearKV):
            kv.ctl_tracker["tree"] = tree
        else:
            kv.tracker["tree"] = tree
    verify_tokens = torch.zeros((embeds.shape[0], tree.rows), device=device, dtype=torch.long)
    verify_buffers = (torch.empty((embeds.shape[0], tree.rows, cfg.hidden_size), device=device, dtype=dt),
                      ftab[:, :, pos:pos + tree.rows].clone())
    draft_top = torch.empty((embeds.shape[0], 7 if adaptive else tree.fanout), device=device, dtype=torch.long)
    embed = self.model.embed_tokens
    de = verify_head.EmbeddingRows.create(_C, embed, device)
    draft_token = torch.empty_like(nt_buf)
    draft_emb = torch.empty_like(h_buf)
    draft_hidden = torch.empty_like(h_buf)
    draft_freqs = ftab[:, :, pos:pos + 1].clone()

    # device state: st = [committed length, MTP KV length]; the caches' prepare / advance are ours now
    st = torch.tensor([pos, mtp_kv.index, 0, 0], dtype=torch.long, device=device)
    for kv in pkv + [mtp_kv]:
        kv._mono_device = True
    ctl = lin[0].ctl
    parity = lin[0].ctl_tracker["parity"]
    ctl_init = torch.tensor([0, parity] + list(range(SPEC_ROWS)) + [0] * (2 * SPEC_ROWS), dtype=torch.int32)
    ctl.copy_(ctl_init.to(device))  # first decode step: nothing to replay (upstream: uncommitted 0 after prefill)
    apos, aseq = attn[0].position, attn[0].seqlen
    verify_pos = torch.zeros((1, tree.rows), dtype=torch.long, device=device)
    # device addresses can have the top bit set: store their two's-complement int64 bits
    kv_ptrs = torch.tensor([p - (1 << 64) if p >= (1 << 63) else p
                            for kv in attn for p in (kv.key.data_ptr(), kv.value.data_ptr())],
                           dtype=torch.long, device=device)
    kv_heads, kv_cap, head_dim = attn[0].key.shape[1], attn[0].key.shape[2], attn[0].key.shape[3]
    children_dev, tree_ctl_dev = _tree_tensors(tree, device)
    no_cand = torch.empty(0, dtype=torch.long, device=device)
    ring = torch.zeros((RING, 32), dtype=torch.long, device=device)
    host_ring = torch.zeros((RING, 32), dtype=torch.long, pin_memory=True)
    best_widths = getattr(q35.best_widths, "__wrapped__", q35.best_widths)  # upstream scoring, our timings
    mode = MODE if MODE in ("sync", "shadow") else "async"
    timed = adaptive

    def embed_rows(tokens):
        if (hasattr(embed, "_v") and embed.weight_lowvram_function is None and len(embed.weight_function) == 0
                and comfy.ops.vbar_above_watermark(embed)):
            with _lock:
                _stats["host_rows"] += 1
            return embed.host_rows(tokens, out_dtype=dt)  # reads the ids on the host (a sync)
        if de is not None:
            out = de.forward(tokens)
            if out is not None:
                return out
        return embed(tokens).to(dt)

    fused_sampling = _sampling.fused_compatible(self) if sampling is not None else False

    def verify_sample(lg):
        s = sampling
        if s is None:
            return lg[0].argmax(dim=-1).unsqueeze(0)
        if fused_sampling:
            out = _sampling.fused_sample(_C, lg[0], s, generator)
            if out is not None:
                return out.transpose(0, 1)
        # upstream's sampling on lg[0].float(); its top-k filter served exactly by topk_rows
        ctx, rows = _sampling.verify_rows(_C, lg[0])
        with ctx:
            out = self.sample_token(rows, s["temperature"], s["top_k"], s["top_p"], s["min_p"], 1.0, [], generator)
        return out.transpose(0, 1)

    def enqueue(s):
        """Queue step s entirely on the device: drafts, verify, sampling, accept; returns its record."""
        nonlocal parity
        ev = [torch.xpu.Event(enable_timing=True) for _ in range(2)] if timed else None
        if ev:
            ev[0].record()
        verify_tokens[:, :1].copy_(nt_buf)
        if dh is None:
            draft_hidden.copy_(h_buf)
        hid = h_buf  # the fused head: a draft's pre-norm hidden feeds the next one directly
        for k in range(depth):
            _C.mtp_prep_draft(st, k, ftab, draft_freqs, mtp_kv.position, mtp_kv.seqlen)
            if dh is not None:
                children = tree.children[k]
                nout = 7 if adaptive else max(tree.fanout, 1)
                top = draft_top if adaptive or tree.fanout > 1 else None
                hid = dh.forward(verify_tokens[:, k:k + 1], hid, draft_freqs, mtp_kv)
                dh.pick(nout, not adaptive, top, draft_ranks[k] if adaptive else None, draft_token, verify_tokens, children)
                continue
            draft_token.copy_(verify_tokens[:, k:k + 1])
            draft_emb.copy_(embed_rows(draft_token))
            n1, r1 = self.mtp(draft_emb, draft_hidden, draft_freqs, mtp_kv)
            draft_hidden.copy_(r1)
            children = tree.children[k]
            nout = 7 if adaptive else max(tree.fanout, 1)
            top = draft_top if adaptive or tree.fanout > 1 else None
            if lm2_on and lm2.pick(n1, nout, not adaptive, top, draft_ranks[k] if adaptive else None, draft_token,
                                   verify_tokens, children):
                continue  # the draft's top-n, rank row, token and children's verify tokens in one pick
            logits = self.logits(n1)[:, -1]
            if adaptive:
                draft_top.copy_(logits.topk(7, dim=-1).indices)
                draft_token.copy_(draft_top[:, :1])
                draft_ranks[k].copy_(draft_top[0])
            else:
                draft_token.copy_(logits.argmax(dim=-1, keepdim=True))
                if tree.fanout > 1:
                    draft_top.copy_(logits.topk(tree.fanout, dim=-1).indices)
            verify_tokens[:, children[0]:children[0] + 1].copy_(draft_token)
            for j, c in enumerate(children[1:], 1):
                verify_tokens[:, c:c + 1].copy_(draft_top[:, j:j + 1])
            if lm2 is not None:  # check mode: the two-stage result against the int8 one
                lm2.check(n1, nout, not adaptive, top, draft_token, verify_tokens, children)
        if ev:
            ev[1].record()
        parity ^= 1
        _C.mtp_prep_verify(st, apos, aseq, verify_pos, tree.positions.reshape(-1), ctl, tree_ctl_dev, parity, rope_offset)
        with vh.capture() if vh is not None else contextlib.nullcontext():
            x, _, _ = self.model.forward(None, embeds=embed_rows(verify_tokens), attention_mask=None, past_key_values=pkv,
                                         position_ids=verify_pos, decode_buffers=verify_buffers)
        lg = verify_logits(x)
        toks = lg.argmax(dim=-1) if sampling is None else verify_sample(lg)
        slot = ring[s % RING]
        rec = {"step": s, "depth": depth, "ev": ev}
        if mode == "shadow":  # upstream's inputs, before the accept kernel changes the state
            rec["shadow"] = (toks[0].tolist(), verify_tokens[0].tolist(), draft_ranks.reshape(-1).tolist() if adaptive else None,
                             st.tolist(), [list(c) for c in tree.children], tree.rows)
        _C.mtp_accept(st, toks.contiguous(), verify_tokens, draft_ranks if adaptive else no_cand, children_dev, tree.rows,
                      depth, x.contiguous(), nt_buf, h_buf, ctl, kv_ptrs, kv_heads, kv_cap, head_dim, slot)
        host_ring[s % RING].copy_(slot, non_blocking=True)
        done = torch.xpu.Event()
        done.record()
        rec["done"] = done
        if mode == "shadow":
            rec["after"] = (st.tolist(), ctl.tolist())
        return rec

    def shadow_check(rec, r):
        toks, dr, cand, st0, children, rows = rec["shadow"]
        path, commit, ranks, row, on_chain, kept = _upstream_walk(children, toks, dr, rec["depth"], cand)
        st1, ctl1 = rec["after"]
        n = int(r[0])
        ok = (n == len(path) and [int(v) for v in r[1:1 + n]] == commit and int(r[9]) == row and bool(r[10]) == on_chain
              and [int(v) for v in r[12:12 + int(r[11])]] == ranks and int(r[20]) == kept
              and st1[0] == st0[0] + len(path) and st1[1] == st0[1] + kept and ctl1[0] == len(path)
              and ctl1[2:2 + len(path)] == path)
        with _lock:
            _stats["shadow_checked"] += 1
            if not ok:
                _stats["shadow_mismatch"] += 1

    last = [None]  # the previously finished step (its verify ends where this step's drafts begin)

    def finish(rec):
        """Process step rec (complete) on the host: tokens, stop, progress, auto-tree statistics."""
        nonlocal steps, draft_ms, verify_ms, widths, depth, children_dev, tree_ctl_dev
        rec["done"].synchronize()
        r = host_ring[rec["step"] % RING].tolist()
        if mode == "shadow":
            shadow_check(rec, r)
        n = int(r[0])
        commit = [int(v) for v in r[1:1 + n]]
        if adaptive:
            steps += 1
            for row in hits:
                for i in range(8):
                    row[i] *= 0.984375
            for level, rank in enumerate(int(v) for v in r[12:12 + int(r[11])]):
                hits[level][rank] += 1
            # device time (upstream's split): this step's drafts / their count, and the previous step's
            # verify .. this step's first draft (verify, lm_head, sampling, accept); both are complete
            prev = last[0]
            if steps > 8 and rec["ev"] and prev is not None and prev["ev"]:
                mix = 1.0 if steps == 9 else 0.125
                d = rec["ev"][0].elapsed_time(rec["ev"][1]) / max(1, rec["depth"])
                v = prev["ev"][1].elapsed_time(rec["ev"][0])
                draft_ms += mix * (d - draft_ms)
                verify_ms += mix * (v - verify_ms)
            if steps % 32 == 0:
                if steps > 8:
                    autotree.note_device_timing(steps, draft_ms, verify_ms)
                best = best_widths(widths, hits, draft_ms, verify_ms)
                if best != widths:
                    widths = best
                    tree.update(q35.tree_spec(widths))
                    depth = tree.chain
                    children_dev, tree_ctl_dev = _tree_tensors(tree, device)
        commit = commit[:max_length - len(ids)]
        stop = next((i for i, t in enumerate(commit) if t in stop_tokens), None)
        if stop is not None:
            del commit[stop + 1:]
        ids.extend(commit)
        update_progress(len(commit))
        last[0] = rec
        with _lock:
            _stats["steps"] += 1

    with _lock:
        _stats["generates"] += 1
    _tls.async_loop = mode == "async"
    _tls.device_loop = True
    try:
        s = 0
        pending = None
        while True:
            if pending is None and (len(ids) >= max_length or ids[-1] in stop_tokens):
                break
            rec = enqueue(s)
            s += 1
            if mode != "async":
                finish(rec)
                continue
            if pending is not None:  # the previous step is processed while this one is already queued
                finish(pending)
                if len(ids) >= max_length or ids[-1] in stop_tokens:
                    break  # `rec` ran past the end: dropped
            pending = rec
    finally:
        console.close()
        _tls.async_loop = _tls.device_loop = False
        torch.xpu.synchronize(device)  # queued steps still read the weights, caches and graphs released below
        comfy.model_prefetch.cleanup_prefetch_queues()
        for kv in pkv + [mtp_kv]:
            kv._mono_device = False
        if dh is not None:
            with _lock:
                _stats["draft_fused"] = _stats.get("draft_fused", 0) + int(dh.calls.item())
        if lm2 is not None:
            calls, bad, checked = lm2.counters()
            with _lock:
                _stats["draft_int4_generates"] += 1
                _stats["draft_int4"] += calls - checked
                _stats["draft_int4_checked"] += checked
                _stats["draft_int4_mismatch"] += bad
    return ids
