import sys, torch
exec(open('/tmp/octave_bench.py').read().split("nb = lambda")[0])
print('\nper-layer us, splits 32 vs 256 (num_q=4 verify | num_q=1 draft)')
for L in (1000, 4000, 8000, 16000, 32000, 70000):
    a = run(L, 4, 32, 4, iters=40)[0]; b = run(L, 4, 256, 4, iters=40)[0]
    c = run(L, 1, 32, 1, iters=40)[0]; d = run(L, 1, 256, 1, iters=40)[0]
    print('kv_len %6d: q=4  32 sp %6.0f  256 sp %6.0f | q=1  32 sp %6.0f  256 sp %6.0f' % (L, a, b, c, d))
