import gzip, json, glob
f = sorted(glob.glob('/tmp/prof/rank0.*.trace.json.gz'))[-1]
d = json.load(gzip.open(f))
ev = sorted([e for e in d['traceEvents'] if e.get('ph') == 'X'], key=lambda e: e['ts'])
marks = [e for e in ev if e['name'].startswith('execute_context_0(0)_generation_1')]
print('decode markers', len(marks))
# start-to-start period between consecutive decode steps (CPU wall clock of the worker)
per = [(marks[i+1]['ts'] - marks[i]['ts']) / 1000 for i in range(len(marks) - 1)]
print('period between consecutive execute_model calls (ms):', ['%.1f' % p for p in per])
print('mean %.1f ms' % (sum(per) / len(per)))
# pick a steady step in the middle and list every event > 150 us in that interval
i = len(marks) // 2
a, b = marks[i]['ts'], marks[i + 1]['ts']
print('\n--- events > 150us between marker %d and %d (offsets ms from the first) ---' % (i, i + 1))
for e in ev:
    if a <= e['ts'] < b and e['dur'] > 150:
        print('  +%7.2f ms  dur %7.2f ms  %s' % ((e['ts'] - a) / 1000, e['dur'] / 1000, e['name'][:90]))
# gaps: time inside the interval not covered by any event of the worker thread
cov = 0
last = a
for e in ev:
    if a <= e['ts'] < b:
        end = e['ts'] + e['dur']
        if end > last:
            cov += end - max(last, e['ts']); last = end
print('\ncovered by profiled CPU events: %.1f of %.1f ms' % (cov / 1000, (b - a) / 1000))
