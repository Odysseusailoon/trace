# validation: t10_search_plus_calc

- episodes: 200, gold = 67.5, overall correct = 6.5%
- MI(first-decision cluster; correct) = 0.000 bits
- big-cluster correct-rate spread = 0.00 -> not locked

| first-decision cluster | n | share | correct | outcomes |
|---|---|---|---|---|
| `search_first` | 200 | 100% | 6% | {'wrong_other': 187, 'correct': 13} |
