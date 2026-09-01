# validation: t4_artist_album_ratio

- episodes: 200, gold = 1.701, overall correct = 30.5%
- MI(first-decision cluster; correct) = 0.568 bits
- big-cluster correct-rate spread = 1.00 -> DECISION-LOCKED

| first-decision cluster | n | share | correct | outcomes |
|---|---|---|---|---|
| `sql:intdiv` | 91 | 46% | 0% | {'intdiv_1.000': 90, 'wrong_other': 1} |
| `sql:other` | 66 | 33% | 39% | {'correct': 26, 'wrong_other': 26, 'intdiv_1.000': 14} |
| `sql:avg_groupby` | 35 | 18% | 100% | {'correct': 35} |
| `sql:from_artist` | 8 | 4% | 0% | {'wrong_other': 7, 'wrongtable_1.262': 1} |
