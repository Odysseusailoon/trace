# validation: t7_avg_track_len_min

- episodes: 200, gold = 6.56, overall correct = 99.5%
- MI(first-decision cluster; correct) = 0.012 bits
- big-cluster correct-rate spread = 0.03 -> not locked

| first-decision cluster | n | share | correct | outcomes |
|---|---|---|---|---|
| `sql:avg_convert_in_sql` | 148 | 74% | 100% | {'correct': 148} |
| `sql:avg_ms_raw` | 37 | 18% | 97% | {'correct': 36, 'wrong_unit_seconds': 1} |
| `sql:other` | 15 | 8% | 100% | {'correct': 15} |
