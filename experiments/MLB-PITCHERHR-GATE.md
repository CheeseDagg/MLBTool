==========================================================================
PITCHER-HR GATE — start-level placebo + shape, on the wide panel
==========================================================================
REAL  n_hold=24332  w_p=0.3  holdout dLL +0.00025  2024 +0.00011  2025 +0.00053  slices 4/4  robust=True

--- GATE 2: START-LEVEL PLACEBO (20 trials, full tune each)
    The baseline is reused across trials ON PURPOSE and the first
    trial asserts it: the baseline uses league, batter, park and
    platoon cells, none of which are keyed on starter identity, so
    an sp permutation provably cannot move it. Re-fitting it 20
    times would only add noise to the comparison.
ship rule fired on noise: 0/20 (0%)
noise dLL >= real (+0.00025): 0/20 (0%)  <- this is the p-value
noise dLL  min -0.00027  median -0.00012  max +0.00003

--- GATE 3: SHAPE
    (a) the effect must be carried by starters whose HR-allowed cell
        is actually populated. A term that pays off just as well on
        pitchers with 40 batters faced as on pitchers with 600 is
        not reading pitcher quality, it is reading something else.
    thin cells   (pit PA < 200)  n= 20345  dLL -0.00001  slices 2/4  robust=False
    medium       (200-600)       n= 23221  dLL +0.00014  slices 2/4  robust=False
    mature cells (600+)          n= 30452  dLL +0.00044  slices 3/4  robust=True

    (b) weight sweep. w_p scales how much of the raw pitcher ratio is
        believed. A real signal degrades gently either side of its
        best value; a grid artifact lives in one cell and dies in the
        neighbours. The shipped value is NOT changed by this sweep —
        these are holdout numbers and picking the best one is a leak.
    w_p 0.15   dLL +0.00015  2024 +0.00008  2025 +0.00030  slices 4/4
    w_p 0.30   dLL +0.00025  2024 +0.00011  2025 +0.00053  slices 4/4
    w_p 0.45   dLL +0.00029  2024 +0.00009  2025 +0.00069  slices 3/4
    w_p 0.60   dLL +0.00027  2024 +0.00001  2025 +0.00080  slices 3/4
    w_p 0.80   dLL +0.00015  2024 -0.00019  2025 +0.00083  slices 2/4
    w_p 1.00   dLL -0.00008  2024 -0.00049  2025 +0.00073  slices 1/4
    w_p 1.40   dLL -0.00092  2024 -0.00144  2025 +0.00012  slices 1/4
