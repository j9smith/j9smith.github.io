+++
date = '2026-06-23T09:19:02+01:00'
draft = true
title = 'Attention'
summary = 'A deep dive into attention.'
+++

Attention fundamentals — QKV, scaled dot-product, complexity
Multi-head attention → MQA/GQA — the memory motivation
KV cache — what it stores, why it matters, memory math
KV cache hierarchy — GPU HBM → CPU → disk, eviction
FlashAttention — IO-aware computation, tiling
PagedAttention — virtual memory analogy, fragmentation problem
RadixAttention / prefix caching — tree structure, cache reuse
Chunked prefill — prefill/decode interference
Disaggregated prefill/decode — separate fleets, why it matters at scale