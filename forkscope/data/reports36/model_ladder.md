# Model ladder: Qwen3-8B vs Qwen3.6-35B-A3B (same tasks, same seeds)

| task | 8B correct | 3.6-35B correct | MI (first-step; correct) | 3.6 dominant first step |
|---|---|---|---|---|
| t4_artist_album_ratio | 30.5% | 28.0% | 0.57 -> 0.27 | sql:other (110/200) |
| t7_avg_track_len_min | 23.5% | 99.5% | 0.02 -> 0.01 | sql:avg_convert_in_sql (148/200) |
| t10_search_plus_calc | 6.5% | 97.0% | 0.00 -> 0.00 | search_first (200/200) |

- t7: capability FIXES the mid-trajectory knowledge fork (the 8B-rare convert-in-SQL path became dominant).
- t4: capability RELOCATES the trap: intdiv extinct; the semantically principled `FROM Artist` opening locks 1.262 at ~99%.
- t10: capability FIXES the decoy lock (6.5% -> 97.0%; the model now computes 19.3/28.6 instead of copying the adjacent 67.3).

Two of three failure modes dissolved under a one-generation capability jump; the semantic decision trap (t4) survived it unchanged. Failure taxonomy predicts which: knowledge/decoy forks are capability-soluble, decision-layer forks are not.
