#!/bin/bash
D=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== CLI exposure (arg_utils):"
grep -n -E "prefix_match_unit|prefix-match-unit|mamba_shared_prefix|mamba-shared-prefix|prefix_cache_retention|prefix-cache-retention" $D/engine/arg_utils.py | head -12
echo "=== uses of enable_mamba_shared_prefix_checkpoint:"
grep -rn "enable_mamba_shared_prefix_checkpoint" $D --include=*.py | grep -v "config/cache.py" | head -12
echo "=== uses of prefix_match_unit:"
grep -rn "prefix_match_unit" $D --include=*.py | grep -v "config/cache.py" | head -14
echo "=== env vars with match unit / shared prefix:"
grep -n -i -E "PREFIX_MATCH|SHARED_PREFIX|MATCH_UNIT" $D/envs.py | head
echo "=== what start_qwen passes today:"
grep -n -E "prefix-match-unit|prefix-cache-retention|PREFIX_MATCH|match.unit|shared-prefix" /app/single-user/start_qwen.sh | head -12
