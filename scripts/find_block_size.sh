#!/bin/bash
# How does vLLM size the attention/mamba block of a hybrid model, and is there a knob for the prefix-cache match granularity?
D=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== options on the cache config:"
grep -n -E "^\s+(mamba_block_size|block_size|prefix_match_unit|cache_hit_alignment_tokens|enable_mamba_shared_prefix_checkpoint|mamba_cache_mode|prefix_cache_retention_interval|mamba_page_size_padded)\b" $D/config/cache.py | head -20
echo "=== where the hybrid block size is chosen:"
grep -rn -E "def (update_block_size_for_backend|_update_block_size|get_hybrid_block_size|unify_kv_cache_spec_page_size|_get_kv_cache_groups_uniform_page_size)" $D --include=*.py | head -10
grep -n "attn_block_size\|mamba_page_size\|attn_page_size_1_token\|block_size = " $D/model_executor/models/config.py | head -30
echo "=== CLI flags mentioning block size / match unit:"
/app/venv/bin/vllm serve --help=all 2>/dev/null | grep -i -E "block-size|match-unit|alignment|prefix-cache-retention|mamba-cache-mode" | head -12
