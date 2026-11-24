cd /src
SP=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
echo "vllm package dir: $SP"
file patches/series patches/spec-decode-attn.patch | head -3
echo "--- series list check:"
bash patches/apply.sh --list >/dev/null && echo "list ok" || echo "list FAILED"
echo "--- apply series:"
bash patches/apply.sh "$SP" 2>&1 | tail -90
echo "apply exit=${PIPESTATUS[0]}"
