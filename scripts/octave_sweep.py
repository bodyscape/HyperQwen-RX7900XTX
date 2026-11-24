import sys, torch
exec(open('/work/octave_bench.py').read().split("nb = lambda")[0])   # reuse run()

print('\nnum_q=4, sweep qgroup x splits  (time us per layer)')
for L in (70000, 125000):
    print('kv_len', L)
    print('  qgrp \\ splits ' + ''.join('%8d' % s for s in (64, 128, 192, 256, 384, 512)))
    for qg in (1, 2, 4, 8):
        row = []
        for sp in (64, 128, 192, 256, 384, 512):
            try:
                t, _ = run(L, 4, sp, qg, iters=25)
                row.append('%8.0f' % t)
            except Exception as ex:
                row.append('%8s' % 'ERR')
        print('  %14d ' % qg + ''.join(row))
print('\nnum_q=1 (drafter steps), splits sweep at 125k')
for sp in (32, 128, 256, 512):
    t, _ = run(125000, 1, sp, 1, iters=25)
    print('  splits %4d: %.0f us' % (sp, t))
print('\nnum_q=2 and 3 (shorter verify), splits 128, qgroup=num_q')
for nq in (2, 3, 4):
    t, _ = run(125000, nq, 128, nq, iters=25)
    print('  num_q %d: %.0f us' % (nq, t))
