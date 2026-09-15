import csv

rows = list(csv.DictReader(open('/TeRed+RATE/code/results/e6e_gnn_baseline.csv')))
ORDER = ['identity+rate', 'tered+dual_naive', 'tered+rate', 'tered+rate_ratio',
         'tered+rate_single', 'tered+none']
for det in ['cosine', 'gnn']:
    for cfg in ORDER:
        rs = [r for r in rows if r['detector'] == det and r['config'] == cfg]
        if not rs:
            continue
        ok5 = sum(1 for r in rs if r['F1_alert@5'] not in ('', None))
        al = [int(r['n_alerts']) for r in rs]
        tn = [int(r['n_alerted_nodes']) for r in rs]
        print(f'{det:7s} {cfg:20s} n={len(rs)} rows_with_@5={ok5} '
              f'n_alerts={al} alerted_nodes={tn}')
