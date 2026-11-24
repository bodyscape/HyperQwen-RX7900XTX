cd /app
cp /tmp/start-rocm-new.sh /app/start-rocm-new.sh
bash -n /app/start-rocm-new.sh && echo syntax-ok
for kv in bf16 octave; do
  echo "== HQ_ROCM_KV=$kv"
  env -u SPEC -u DRAFT_TOKENS -u EXTRA_ARGS -u DRAFT_ATTN_BACKEND -u MAX_LEN HQ_ROCM_KV=$kv PRINT_ARGV=1 bash /app/start-rocm-new.sh 2>&1 \
    | tr ' ' '\n' | grep -E -A1 -e '^--(attention-backend|kv-cache-dtype|speculative-config|max-model-len)$' | grep -v -e '^--$' | paste -sd' '
done
rm -f /app/start-rocm-new.sh
