======================================================================
MLB HR MARKET ANCHOR — blend published HR% toward the de-vigged price
run 2026-08-04, on mlb/data/hr_graded.csv (691 graded rows, 163 priced)
======================================================================

MOTIVATION (measured, not theorized): the claimed-edge curve on the graded
ledger is a winner's-curse signature — hit rate DECAYS as the model's claimed
EV over the book rises:

    book likes it more (EV<=0)   n= 56   16.1% hit
    claimed edge 0-10%           n= 11    9.1%
    claimed edge 10-20%          n= 18   16.7%
    claimed edge 20-30%          n= 18    0.0%
    claimed edge 30%+            n= 60    5.0%

The model's biggest disagreements with the book are its biggest errors.

DE-VIG: the ledger logs the Yes side only, so devig_two_way is unavailable.
Single-sided haircut p_market = implied(book_price) / (1 + hold), hold = 11.5%
(midpoint of the 8-15% Yes-side prop-hold band documented in mlb_hr.py).
Market-only Brier is insensitive across the band (.0859 at 8%, .0853 at 11.5%,
.0848 at 15%) — every value in it beats the model, the midpoint isn't the story.

WALK-FORWARD BY DATE (train = all earlier priced dates; blend
logit(p) = w*logit(model) + (1-w)*logit(market), w refit each date on a grid):

    date        n  hits  w_fit  Brier model  Brier market  Brier blend
    2026-07-07  30   3     -      .08991       .08391         -      (no train yet)
    2026-07-08  31   8    0.00    .19082       .18615       .18615
    2026-07-09  33   3    0.00    .09973       .08610       .08610
    2026-08-01  35   2    0.00    .06613       .06184       .06184
    2026-08-02  34   0    0.00    .02724       .01777       .01777
    pooled     133               .09359       .08557       .08557
    clean subset (>= 2026-08-01, re-pulled prices; pre-08-01 prices may be
    wrong-game):  n=69            .04697       .04012       .04012

STABILITY OF w: fitted model weight is 0.00 at every walk-forward step, on
every leave-one-date-out fold, on the clean subset alone, and on 94% of 400
bootstrap resamples (90% CI [0.00, 0.05]). Brier AND log-loss are monotone
increasing in model weight (LL .3042 at w=0 -> .3362 at w=1). The market wins
outright — this is not "blend a little toward the book", it is "the book's
number is better than ours everywhere both exist".

BLEND'S CLAIMED-EDGE CURVE vs THE MODEL'S: at w=0 the published number IS the
de-vigged market, so the published claimed edge vs the book is never positive —
the pathological 20-30% / 30%+ bands (0.0% and 5.0% hit) cannot be published
any more. The MODEL's claimed edge (ev_pct) is deliberately still computed off
the model's own number so the winner's-curse curve keeps grading it.

SHIPPED (2026-08-04):
  * mlb_hr.py: MKT_HOLD=0.115, MKT_W=0.0 (measured, see above). Priced rows
    publish anchor_prob(model, book_price); the model's calibrated number stays
    on the row and in both CSV logs as hr_model. fair repriced off the
    published number. ev_pct stays the model's claim (feeds ev_curve).
    Unpriced rows: pure model, untouched. Refit: python mlb_hr.py
    --refit-anchor (rerun monthly / every ~100 new priced rows; raise MKT_W
    only on a walk-forward win).
  * mlb_grade.py: GCOLS + hr_model (schema gen 19); panel anchor_tier grades
    published (anchored) vs model Brier head-to-head on anchored rows;
    top_tier and agree_tier read the MODEL number so they remain statements
    about the model, not rankings of the book's prices.
  * mlb_livelevel.py: level k fits against hr_model where present (k
    multiplies the model, so fitting it on market-published rows would bias
    k toward 1).
  * index.html: verdict box notes the anchor; anchored HR% cells show
    "mkt · model X%" with a tooltip.

CAVEATS: 163 priced rows, 16 hits, 5 dates. The sample is 2sd cold even
against the market (expected ~26 hits) — selection into the board plus
variance. Pre-2026-08-01 prices may be wrong-game (date filter added
2026-08-03); the clean subset agrees with the full sample on both the winner
and w=0. The anchor is now on the ledger (hr_model), so anchor_tier grades it
forward from here.
