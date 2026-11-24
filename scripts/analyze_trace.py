import gzip, json, glob, collections, sys
f = sorted(glob.glob('/tmp/prof/rank0.*.trace.json.gz'))[-1]
d = json.load(gzip.open(f))
ev = [e for e in d['traceEvents'] if e.get('ph') == 'X']
print('events', len(ev), 'file', f)
cats = collections.Counter(e.get('cat') for e in ev)
print('categories:', dict(cats))

# top-level step markers
steps = sorted([e for e in ev if e['name'].startswith('execute_context')], key=lambda e: e['ts'])
print('\nexecute_context markers:', len(steps))
for e in steps[:40]:
    print('  %-55s dur %8.2f ms' % (e['name'], e['dur'] / 1000))

def agg(cat_filter, top=18):
    c = collections.defaultdict(lambda: [0, 0.0])
    for e in ev:
        if cat_filter(e):
            c[e['name']][0] += 1
            c[e['name']][1] += e['dur']
    return sorted(c.items(), key=lambda kv: -kv[1][1])[:top]

print('\nCPU-side runtime calls (cuda_runtime / hip), by total time:')
for n, (k, t) in agg(lambda e: e.get('cat') in ('cuda_runtime', 'cuda_driver')):
    print('  %-50s n=%5d total %9.2f ms  avg %8.1f us' % (n[:50], k, t / 1000, t / k))

print('\nGPU kernels / memcpy / memset by total time:')
for n, (k, t) in agg(lambda e: e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')):
    print('  %-60s n=%5d total %9.2f ms  avg %8.1f us' % (n[:60], k, t / 1000, t / k))

gk = [e for e in ev if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
if gk:
    t0 = min(e['ts'] for e in gk); t1 = max(e['ts'] + e['dur'] for e in gk)
    busy = sum(e['dur'] for e in gk)
    print('\nGPU window %.1f ms, busy %.1f ms (%.0f%%), kernels %d' % ((t1 - t0) / 1000, busy / 1000, 100 * busy / (t1 - t0), len(gk)))

print('\nTop CPU ops (cpu_op / python_function / user_annotation) by total time:')
for n, (k, t) in agg(lambda e: e.get('cat') in ('cpu_op', 'user_annotation', 'python_function'), 22):
    print('  %-60s n=%5d total %9.2f ms  avg %8.1f us' % (n[:60], k, t / 1000, t / k))
