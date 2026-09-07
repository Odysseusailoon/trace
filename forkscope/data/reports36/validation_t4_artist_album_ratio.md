# validation: t4_artist_album_ratio

- episodes: 200, gold = 1.701, overall correct = 28.0%
- MI(first-decision cluster; correct) = 0.266 bits
- big-cluster correct-rate spread = 0.49 -> not locked

| first-decision cluster | n | share | correct | outcomes |
|---|---|---|---|---|
| `sql:other` | 110 | 55% | 50% | {'correct': 55, 'wrongtable_1.262': 55} |
| `sql:from_artist` | 90 | 45% | 1% | {'wrongtable_1.262': 89, 'correct': 1} |
