# Expected-innings experiment — 2026-07-27

Trigger: Jack Perkins projected exp_ip 7.0 (pinned cap), λ 7.84, #1 on K board.
No pitcher in the 2,448-start 2025 validation sample earned even a 6.4 projection.

## Tested (walk-forward on kfactors_dataset.json: tune Apr–May, verdict June)
- Shrink start-length toward league mean (tau grid 0.5–8, cap grid 25.5–29.4 BF): **NULL.**
  Train winner (tau=0.5 cap=25.5) LOST on June holdout (−0.00094/start, worse both halves).
- Recency windows (last 3/5/10): worse than season mean on train.
- Subgroups: shrinkage actively HURTS short-leash arms (−0.04 to −0.18/start, n=60) —
  openers/5-inning guys are genuinely short; the raw average is real signal.
  Helps the few-starts-AND-long profile (+0.03) but n=2. Not a formula problem.

## Root cause found instead: INPUT bug (shipped fix)
Runner fed expected_ip(total season IP, games started). Total IP includes RELIEF
innings; dividing by GS only inflates swingmen — exactly the Perkins profile.
Second bug on the same line: statsapi innings are baseball notation ('45.1' =
45⅓), parsed as decimal 45.1.

Fix: game-log pull → mean IP over ACTUAL STARTS drives exp_ip (fail-soft to old
season-line math); ip_float() parses .1/.2 as thirds; K rate still uses ALL
innings (relief Ks are real rate evidence). Selftests: swingman guard, byte-
identical pure-starter and fallback paths.

Verdict: formula unchanged (validated); estimator inputs fixed.
