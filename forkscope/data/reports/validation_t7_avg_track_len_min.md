# validation: t7_avg_track_len_min

- episodes: 200, gold = 6.56, overall correct = 23.5%
- MI(first-decision cluster; correct) = 0.023 bits
- big-cluster correct-rate spread = 0.00 -> not locked

| first-decision cluster | n | share | correct | outcomes |
|---|---|---|---|---|
| `sql:avg_ms_raw` | 197 | 98% | 23% | {'wrong_unit_seconds': 150, 'correct': 45, 'wrong_other': 2} |
| `sql:avg_convert_in_sql` | 2 | 1% | 100% | {'correct': 2} |
| `sql:other` | 1 | 0% | 0% | {'wrong_unit_seconds': 1} |
