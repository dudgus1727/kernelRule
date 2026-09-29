# a6000 규칙 손편집 — 16회 장부 (D-188)

> **재현** 규칙 하나: `python3 -m experiments.a6000_probe --rule docs/artifacts/a6000-probe/rules/<tag>.json --out OUT.json`
> 이 표: `python3 -m experiments.a6000_ledger` (계산 0회)

숫자 읽는 법: **1.00 = 가장 빠른 설정을 정확히 고름, 1.08 = 8% 느림.** 작을수록 좋다.

- **고르는 값(inner-CV)** — 학습 형상 안에서만 잰 값. 반복 사이에서 무엇을 남길지는 이것으로 정했다.
- **보고 값(홀드아웃)** — 네 fold 의 시험 형상 65개. 벤더는 **1.0815**.

## 기준선 — 원래 규칙(c2 루프가 만든 것)을 같은 절차로

| 시드 | inner-CV | 홀드아웃 | 벤더 넘은 fold | 쪼개기 문제 적중 |
|---|--:|--:|--:|--:|
| s0 | 1.1177 | 1.1149 | 0/4 | 13/31 |
| s1 | 1.0976 | 1.1023 | 1/4 | 8/31 |
| s2 | 1.1013 | 1.1460 | 0/4 | 5/31 |
| s3 | 1.0868 | 1.1258 | 0/4 | 10/31 |

## 16회

| # | 라운드 | 무엇을 | 대조군 | 판정 | inner-CV | 홀드아웃 | 벤더 넘은 fold | 쪼개기 적중 |
|--:|--:|---|:-:|---|--:|--:|--:|--:|
| 1 | 1 | H01 L1 split-K fix plus r09 large-grid branch, with w0 scaled fr |  | inconclusive | 1.0766 | **1.0759** | 3/4 | 25/31 |
| 2 | 1 | H02 R7 flat rule + gated split-K quadratic: refuted (the split-K |  | refuted | 1.0848 | 1.0905 | 1/4 | 12/31 |
| 3 | 1 | H06 Control: a one-direction split-K term in the L1 gate loses e | ○ | confirmed | 1.1066 | 1.1131 | 0/4 | 14/31 |
| 4 | 1 | H13 Library-only rule with split-K terms inside roofline branche |  | inconclusive | 1.1142 | 1.1108 | 1/4 | 21/31 |
| 5 | 2 | H18 A 1-CTA penalty inside the split-K gate removes the 1-CTA pi |  | inconclusive | 1.0723 | **1.0753** | 3/4 | 27/31 |
| 6 | 2 | H19 Over-split guard outside the split-K gate fails and makes th |  | refuted | 1.0884 | 1.1052 | 1/4 | 13/31 |
| 7 | 2 | H20 H01b recipe on the s1 base with formula constants: it transf |  | inconclusive | 1.0771 | **1.0710** | 3/4 | 27/31 |
| 8 | 2 | H07b Control for C2: the split-K quadratic behind the textbook ga | ○ | confirmed | 1.1033 | 1.0989 | 1/4 | 13/31 |
| 9 | 3 | H21 Graded large-grid ramp without the guard: fixes the big clas |  | inconclusive | 1.0848 | 1.1202 | 1/4 | 18/31 |
| 10 | 3 | H22 H20's formula-constant recipe on the s3 and s2 bases: big ho |  | inconclusive | 1.0806 | 1.0826 | 2/4 | 24/31 |
| 11 | 3 | H23 Margin-based bowl strength: works on the s1 base (Run B ties |  | refuted | 1.0775 | **1.0697** | 3/4 | 28/31 |
| 12 | 3 | H24 Control for C3: split-only placebo axes in H18A's large-grid | ○ | refuted | 1.0821 | **1.0796** | 2/4 | 17/31 |
| 13 | 4 | H25 Control: renaming the weights (identical objective) moves in | ○ | confirmed | 1.0691 | **1.0602** | 3/4 | 26/31 |
| 14 | 4 | H26 A global split-K traffic penalty cuts over-splitting but mov |  | refuted | 1.0844 | 1.1068 | 2/4 | 24/31 |
| 15 | 4 | H27 The start-point guard removes H22's inner-CV loss on s3 fold |  | inconclusive | 1.0673 | 1.0993 | 0/4 | 26/31 |
| 16 | 4 | H28 Per-fold inner-CV choice of base: both composites reproduce  |  | refuted | 1.0550 | **1.0563** | 3/4 | 27/31 |

굵은 홀드아웃 = 벤더보다 낮음. ⚠️ 결과끼리 **0.01 안쪽은 비김**이다 (13번 대조군: 가중치 이름만 바꿔도 0.01 흔들린다).

## 한 회씩 — 무엇을 바꿨고 무엇을 알았나

### 1. H01 — L1 split-K fix plus r09 large-grid branch, with w0 scaled from train

- **시험한 원인**: C1 + C3 are additive: the L1 split-K sweet-spot term (C1) and a large-grid tile-choice branch (C3) fix different shape classes and should not interfere.
- **바꾼 것**: I kept each fold's L1 rule unchanged (its code, starting weights and extra features). Before `return s` I added r09's large-grid branch: when the shape is compute-bound and has a lot of output per SM (roofline_ratio >= 1 and output_work_per_sm > 200000), two extra terms apply, one on CTA residency scarcity and one on register pressure. Each new term's starting weight is set, from each fold's train shapes, to about the size of the rule's existing score spread within a shape, so the terms can actually change picks (C5). The sign was also chosen on train. There were two runs. Run a sized the terms with the plain standard deviation of the base score. Run b used a robust spread (IQR), which ignores spill outliers.
- **돌린 것**: H01a_std (w0 = train sign x std ratio): inner 1.0875 · 홀드아웃 1.0786 / H01b_robust (w0 = train sign x robust-std ratio): inner 1.0766 · 홀드아웃 1.0759
- **판정**: inconclusive — Selected run by inner-CV: H01b (inner 1.0766, against 1.0875 for H01a). H01b has pooled holdout 1.0759 and beats the vendor in 3 of 4 folds. L1 alone had 1.0834 and 2 of 4, and the vendor pools to 1.0815. So it meets the task target, but the 0.0056 margin over the vendor is inside the about 0.01 noise. Confirm criteria: inner-CV below L1's 1.0869 is met (1.0766). Large-grid class at most 1.08 is met (1.060, down from L1's 1.142). Hungry class at most 1.075 is met, just (1.074, against L1's 1.068). Pooled holdout at most 1.0715 is missed (1.0759). Neither refute condition happened: large-grid is not above 1.12 and hungry is not above 1.09. So the verdict is inconclusive.

Additivity mostly holds. Per-class log gap against the vendor, per 65 shapes, going from L1 to H01b: large-grid +0.0140 to +0.0037, hungry -0.0007 to +0.0019, mid-easy +0.0079 to +0.0088, M<=32 -0.0194 to -0.0195. The large-grid gain (0.0103) is much bigger than the small loss on the split-K shapes, so the two fixes stack. H01a gives the same picture: large-grid 1.056, hungry 1.069, pooled 1.0786.

Three things keep it above the 1.0715 target.
(1) The gate owps > 2e5 is too wide. Shapes caught in the swizzle trap are rescued: f2 8192x11008x4096 goes from 1.330 to 1.033, f3 8192x12288x4096 from 1.334 to 1.030, f3 8192^3 from 1.436 to 1.037. But the 2048-row large-grid shapes are hurt: f2 2048x11008x4096 goes from 1.013 to 1.101 and f3 2048x12288x4096 from 1.013 to 1.087, because there the 2-CTA 128x128 tile with swizzle 1 and split_k 2 is already optimal.
(2) Fold 1 is untouched (1.137, L1 1.136). Its losses come from mid-easy shapes that neither cause covers: 512x512x512 at 1.818 and 1024x4096x128 at 1.385.
(3) Hungry shapes regress slightly under refit.

For C3, the branch works through a tile choice, not through residency. On 3 of 4 folds the train data wanted a negative weight on resident_block_scarcity, which favours 1 CTA per SM (128x256 or 256x128). Its swizzle-1 variant is close to optimal, while the 2-CTA 128x128 tile with swizzle 1 is the trap. This is the opposite of the naive residency physics in C4. Fitted weights for the new terms in H01b: f0 [-2.43, -7.10], f1 [+99.5, -143.4], f2 [-17829, -31561], f3 [-31.5, -13.8].
- **상수의 출처**: - **L1 part:** code, w0 and extra_features are copied unchanged, per fold, from an/logs/rule_L1_sweetspot.json. Its gates (33.2131, 32.5, 234057, 249661, 1.32805) were derived from train by the earlier analyst.
- **Gate `p.roofline_ratio >= 1 and p.output_work_per_sm > 200000`:** copied verbatim from r09 (an/table/rules/r09_evolved_s0_regimefix.json) as the hypothesis says. roofline 1 is the physical compute/memory boundary. 200000 is r09's own constant and I did not re-derive it.
- **New w0 magnitudes** (scale.py, it/r1_H01/scale.json): computed on each fold's TRAIN shapes only, via _splits(t, fold=f, k=4, design='nkband').train.shapes, restricted to train shapes inside the gate (8/8/6/5 shapes). The formula is median over those shapes of the within-shape spread of the L1 base score at L1 w0, divided by the median within-shape std of the term (0.2494 for resident_block_scarcity, 0.2239 for register_file_pressure).
  - H01a uses the std of the base score: 14.72, 22.75, 190060 and 4.997 for f0 to f3.
  - H01b uses IQR/1.349 of the base score: 0.597, 20.27, 5493 and 4.897.
- **Signs:** chosen per fold on frozen-L1-w0 TRAIN regret (sweep.py, scale.json sign_eval). I took the best of the four nonzero sign combinations at the std-ratio magnitude, and broke ties with the single-term train sweep. Result: f0 (-,-), f1 (+,-), f2 (-,-), f3 (-,-).
- **Final new w0:**
  - H01a: f0 [-59.02, -65.73], f1 [91.23, -101.59], f2 [-762103, -848677], f3 [-20.04, -22.31]
  - H01b: f0 [-2.39, -2.67], f1 [81.30, -90.53], f2 [-22028, -24530], f3 [-19.64, -21.87]
- No holdout shape or holdout result was used for any constant. The run was chosen by inner_cv_gm.

### 2. H02 — R7 flat rule + gated split-K quadratic: refuted (the split-K block barely helps inside its gate, and refitting the shared weights hurts shapes outside it)

- **시험한 원인**: C1 on top of C3/C4. The test: if the split-K choice gets an interior-optimum term (a quadratic in log2 of the K-loop iterations per serial split), R7's flat rule should pick split_k 3-4 on hungry shapes and keep its gains on easy shapes.
- **바꾼 것**: I took the 16-term R7 rule and added one block that only applies to shapes in the split-K band (large total flops, modest output work per SM). The block is a U-shaped penalty on log2 of how many K-loop iterations each split runs. It is centred near 45 iterations (2^5.5), so the rule prefers a middle number of splits over none or the maximum. The gate thresholds are set per fold from train shapes. The starting weights are R7's own train-fitted weights plus two block weights (-11b, b), with b large enough that the block changes train picks.
- **돌린 것**: A_c5 (b = C5 std-ratio: f0 3.015, f1 6.788, f2 3.640, f3 5.589): inner 1.0887 · 홀드아웃 1.1015 / B_c5div10 (b = C5 std-ratio / 10: f0 0.302, f1 0.679, f2 0.364, f3 0.559): inner 1.0848 · 홀드아웃 1.0905
- **판정**: refuted — Selected run B has holdout 1.0905, which is worse than both R7 (1.0801) and vendor (1.0815). It beats vendor in 1 of 4 folds. It picks split_k >= 3 on only 12 of the 31 hard shapes, and hungry gm is 1.0949. Both refute conditions hold (hungry >= 1.085 and sk>=3 <= 16/31). No confirm criterion is met apart from inner-CV below 1.1007, and that comparison is not fair (see the last point). Run A has the stronger block and is worse: holdout 1.1015, 13/31, easy 1.1176.

Per class for B vs R7 (m1_classgap classes; my wrapper is it/r1_H02/classgap.py):
- hungry: 1.0949 vs 1.0901 (L1: 1.0680)
- large-grid: 1.0430 vs 1.0399 (the confirm target was <= 1.05)
- M<=32: 1.0828 vs 1.0433
- mid-easy: 1.1210 vs 1.1203

Where the damage comes from (holdout split by each fold's gate):
- Inside the gate (33 shapes, 29 of them hard): R7 1.0917, A 1.0813, B 1.0878. The block helps only 0.004-0.010, which is at the noise level.
- Outside the gate (32 shapes, where the block is off): R7 1.0683, A 1.1227, B 1.0932.
- The block adds nothing outside the gate, so that loss comes from refitting R7's 16 weights together with the block. A flat rule has one weight set for all shapes, and the fit gives up easy shapes to serve the gated ones.

Train-only diagnostic (holdout not read): with R7's fitted weights frozen, the quadratic never lowered train regret at any centre from 3 to 7 or any scale from 0.01 to 300.
- It does push picks to split-K. For example, on f1 at b=30, 23 of 28 gated train shapes pick sk=3, but train regret rises from 1.0734 to 1.0902.
- The centre is right on average: at the optimum, L has median 5.42. R7 under-splits: at its pick, L has median 6.0-6.4.
- But the optimum's L ranges from 4.0 to 7.4 across shapes. L also changes with tile_k, not only split_k. So one fixed centre forces wrong tile/split combinations.

What this means for C1: on R7 the missing interior-optimum axis is not the binding problem. R7's per-shape split axes (split_overhead_fraction, log_quantized_work_ratio) already carry most of the split trade-off. L1's 26/31 came on the c2-s0 base, whose weights are split by shape branches. L1 did not beat vendor either: holdout 1.0834 vs 1.0815.

The inner-CV gain (1.0848 vs R7's 1.1007) is not comparable. w0 was R7's fit on the full fold train, so every inner fit starts from weights that already saw its held-out quarter. R7's own inner-CV started from its generic w0. The holdout contradicts that gain.

Deviation: fold 2's L1 gate uses a different axis pair (output_work_per_sm and roofline_ratio). I rederived a fold-2 gate in the hypothesis's form from fold-2 train (see provenance).
- **상수의 출처**: - **Gate, folds 0, 1, 3:** reused L1's train-derived gates from an/logs/fold_gates.json. These came from s19_foldgates.py: on that fold's train shapes, a balanced-accuracy scan of 'fastest sk>=3 beats fastest sk<=2', with thresholds at midpoints between consecutive train values. f0 is log_flops >= 33.2131 and output_work_per_sm < 234057; f1 and f3 are 32.5 and 234057. My own restricted scan on train reproduced these exactly (train BA 0.931/0.904/0.923).
- **Gate, fold 2:** L1's gate there was output_work_per_sm < 249661 and roofline_ratio >= 1.32805, which is not the required form. I reran the same s19 procedure on fold-2 train, restricted to the pair (log_flops >=, output_work_per_sm <). Result: T1 = 33.5056, T2 = 249661, train BA 0.917.
- **w0[0:16]:** R7_compact16.PROBE.json per_fold[f].w, the official probe fit of R7 on fold f's train shapes only.
- **w0[16:18] = (-11b, b):** the centre 5.5 comes from the hypothesis and L1's train sweep (s25).
- **b:** the C5 recipe. For each fold it is the median over gated train shapes of std_within_shape(frozen R7 score) / std_within_shape(L^2 - 11L). Values: f0 3.015, f1 6.788, f2 3.640, f3 5.589. These are run A's values; run B uses the same divided by 10.
- **Not read:** no holdout shape or holdout result went into any constant. The train-only sweeps are in it/r1_H02/prep.log and diag_center.py.

### 3. H06 — Control: a one-direction split-K term in the L1 gate loses everything L1 gained

- **시험한 원인**: Control for C1. On a6000 the best split_k is in the middle (3-4), so a rule needs a term with a bowl-shaped (interior) optimum. A term that only pushes one way should fail.
- **바꾼 것**: I started from the analyst's L1 rule (c2 evolved s0 plus a bowl-shaped term in log2 of K-loop iterations per split, active only inside a gate chosen on train shapes). I deleted the squared line, so only the straight-line term is left, and removed its weight so w0 has no gaps. Everything else is unchanged: the gate, the base rule and the per-fold scale. Run A starts the linear weight at -b, as the hypothesis specifies. Run B starts it at +b, which tests the other one-way direction ("split more"). The CMA fit can flip the sign in either run.
- **돌린 것**: A: linear only, w0=-b (faithful): inner 1.1066 · 홀드아웃 1.1131 / B: linear only, w0=+b (sign-flipped variant): inner 1.1079 · 홀드아웃 1.1137
- **판정**: confirmed — All three confirm conditions hold, so C1 is supported: the interior (bowl-shaped) optimum is what made L1 work.

- **Condition 1, hungry class ≥ 1.11:** m1_classgap gives 1.128 for the hungry class (31 shapes whose best split_k is 3 or more) in both runs. L1 gets 1.068, vendor 1.069, and the unmodified base c2_evolved_s0 gets 1.127.
- **Condition 2, pooled holdout ≥ 1.10:** A 1.1131, B 1.1137. L1 is 1.0834, base 1.1149, vendor 1.0815.
- **Condition 3, picks go to the ends:** they collapse to split_k=1. On the 31 hard holdout shapes, run A picks sk=1 on 17 (sk_best 3 → 1: 9, 4 → 1: 6, 6 → 1: 2). L1 picks sk=1 on only 3 of them. sk≥3 was picked on 14/31 hard shapes (L1: 26/31). Hard-shape gm is 1.128 (L1: 1.068).
- **No sweet spot for the fit to find:** the CMA fit gave different signs to the linear weight by fold (A: f0 -1932, f1 +11.6, f2 -4.6e5, f3 +4.2; B: f0 +3.8, f1 +13.2, f2 -2.7e6, f3 +5.0). Whichever way it points, the result is the same, within 0.001 of the base rule. A one-direction term can only say "split more" or "split less", so the fit either leaves it alone or sends shapes to sk=1.
- **Cost of deleting the squared line:** about +0.020 inner-CV (1.0869 → 1.1066) and +0.030 pooled holdout (1.0834 → 1.1131). This is well above the ~0.01 noise.
- **Where the loss sits:** per-class gaps match the base rule in every class. The hungry class moves -0.0007 → +0.0253 in pooled log gap, while M≤32, big-grid and mid-easy barely move.
- **Fold detail:** f0 1.0775 → 1.1508, f2 1.0581 → 1.0967, f3 1.0637 → 1.0681. f1 is identical to L1 (1.1364) because L1's squared term did not change any fold-1 holdout pick.
- **Beat vendor in 0 of 4 folds** (L1: 2/4).
- **A versus B is a tie:** inner 1.1066 vs 1.1079, well inside noise. A is selected by inner-CV.
- **상수의 출처**: I introduced no new constants.

- **Gates** are copied from rule_L1_sweetspot.json, which the analyst chose on train shapes (an/logs/fold_gates.json):
  - f0: log_flops>=33.2131 and output_work_per_sm<234057
  - f1 and f3: log_flops>=32.5 and output_work_per_sm<234057
  - f2: output_work_per_sm<249661 and roofline_ratio>=1.32805
- **Per-fold scale b** is the magnitude of L1's linear w0, which came from the train-only scale sweep in an/logs/s25_sweetspot.py: f0 1100, f1 12, f2 3.3e6, f3 11. Run A uses w0 = -b; run B uses +b, the same magnitude with the sign flipped.
- **All other w0 values** are the c2_evolved_s0 loop weights as carried in L1. Every weight is refit by the official CMA procedure on each fold's train shapes.
- **Build script:** it/r1_H06/build.py deletes the np.square line and drops the last w0 entry.

### 4. H13 — Library-only rule with split-K terms inside roofline branches (regime_free, all 4 folds)

- **시험한 원인**: C1 in the rules framing: the sign of the wave_count_deficit (wcd) weight should differ by regime. Also C2: does a roofline gate stand in for the split-K need band P?
- **바꾼 것**: I copied the analyst's candidate an/rules/cand/regime_free.json word for word. It has one code for all folds, uses library axes only, and has 18 weights. It splits on p.roofline_ratio < 1, the memory-bound versus compute-bound point, and gives each branch its own split-K terms: wcd, serial_reduction_dependency and parallel_reduction_traffic_fraction, plus a few tile and occupancy terms. Signs are left free. Earlier it had only been measured on fold 0; this time it went through the official probe on all 4 folds. For the second run I kept the same function and starting weights but rescaled each weight per fold so that every term has unit within-shape spread on that fold's TRAIN shapes (C5). This tests whether evenly scaled CMA steps fit better.
- **돌린 것**: main (regime_free verbatim): inner 1.1142 · 홀드아웃 1.1108 / std (same function, per-fold unit-within-shape-sd reparametrisation, keep start): inner 1.1339 · 홀드아웃 1.1279
- **판정**: inconclusive — Inner-CV picks the main run (1.1142 vs 1.1339 for std). Main run: pooled holdout 1.1108 against vendor 1.0815. It beats vendor in 1 of 4 folds, picks split_k >= 3 on 21 of 31 hard shapes, and scores hard 1.051 / easy 1.168.

Checked against the hypothesis:
- Confirm needs holdout < 1.10. It fails at 1.1108.
- The other two confirm conditions pass. On hard shapes sk >= 3 is picked 21/31 (needs >= 20). The wcd sign is negative in the memory-bound branch and >= 0 in the compute-bound branch in 3 of 4 folds: f0 -3.31/+0.70, f2 -0.12/+1.78, f3 -1.30/+1.54. f1 goes the other way (+0.08/-0.14), but both of its weights are close to zero.
- Refute needs holdout >= 1.1149 or 0/4 folds. 1.1108 is below 1.1149 and 1 fold beats vendor, so it is not refuted. The gap to the threshold is 0.004, well inside the 0.01 noise, so the rule effectively ties the c2 s0 baseline.

So the verdict is inconclusive. The mechanism part is supported; the headline number is not.

What this says about C1: putting the split-K terms inside a regime branch does fix the hungry class. By class (m1_classgap copy), C_hungry sk>=3 (n=31) scores 1.051 against vendor 1.069. That is the best of any probe so far (L1 1.068, R7 1.090), and hard_gm 1.051 is the best hard number recorded. In most folds the fitted signs follow the regime story. In effect size, though, wcd is a minor term in the compute branch: about 0.05 to 0.14 within-shape sd units, against about 0.3 to 0.8 for serial_reduction_dependency.

What this says about C2: the roofline gate is the wrong gate. Split-K pressure leaks into small, low-flop shapes and they over-split badly:
- 1024x1024x1024 (fold 2, compute branch, log_flops 31) picks sk=16 for a regret of 3.636 (vendor 1.03).
- 512x512x512 (fold 1) picks sk=8 for 1.818.
- 128x4096x1024 (fold 2, memory branch, where the wcd weight is about 0) picks sk=4 for 1.826.

All three sit outside band P (log_flops < 32.5). The D_mid_easy class scores 1.322 against vendor 1.102, which adds +0.036 to the pooled log gap. Large-grid shapes are still lost as well (B 1.162 vs 1.032, C3). A post-hoc illustration only, not a result: with those 3 shapes at vendor regret the pooled number would be 1.0702. The loss is concentrated in over-splitting outside P, not in the hungry band.

The std variant did worse (inner 1.134, holdout 1.128, 0/4 folds, sk>=3 13/31). With balanced steps CMA wandered to very large weights, such as instruction_density at about -1e4 to -1e5. The compute-branch wcd weight went negative in 4 of 4 folds, so the fit reverted to the anti-split outcome C1 describes. Pooled over-fitting shows up here as larger step sizes, not better fits.

The official f0 numbers (train 1.0975, holdout 1.0544) exactly reproduce the analyst's earlier fold-0-only fit, because the seeds are deterministic.
- **상수의 출처**: The branch threshold p.roofline_ratio < 1 is physics: 1 is the ridge point that separates memory-bound from compute-bound. All other code and the w0 values (0.5, 10, 0.1, 1, ..., -1 for the memory-branch wcd, 0.5 for the compute-branch wcd) were copied unchanged from an/rules/cand/regime_free.json. They are hand-set starting points of order 1, not read from any data. The fitted weights come from the probe's own CMA run on each fold's train shapes.

The std variant's scale factors c_i come from each fold's TRAIN shapes only, via _splits(t, fold=f, k=4, design='nkband').train.shapes. For each weight, term_i = score(w=e_i) - score(w=0) is computed over each train shape's candidates, using config features only and no times. c_i = 1 / (mean within-shape sd over the train shapes where the term varies). The code is rewritten w[i] -> (c_i*w[i]), with w0' = w0/c_i. The script is it/r1_H13/stdize.py, and the c values are stored in rule_std.json per_fold[*].c.

### 5. H18 — A 1-CTA penalty inside the split-K gate removes the 1-CTA picks and fixes the hungry class, but the pooled holdout does not move (inconclusive)

- **시험한 원인**: C4 inside band P. The test: once split_k is roughly right (H01b), is the tile's residency class (1 CTA/SM against the 2-CTA 128x128x32 optimum) the binding error on hungry shapes?
- **바꾼 것**: I started from H01b and kept each fold's code, starting weights and feature files. I added one line inside each fold's existing split-K gate: a penalty on configurations where only one CTA fits per SM. Run A uses the indicator single_resident_cta. Run B uses the library axis resident_block_scarcity (1/CTAs per SM). The starting weight for the new term is taken from that fold's training shapes only. Its size is chosen so the term moves the score about as much as the base rule's typical within-shape spread. Its sign is whichever gives lower training regret with the base frozen, and +1 when neither sign helps. After that, the official probe refits every weight as usual.
- **돌린 것**: A_single_resident_cta: inner 1.0723 · 홀드아웃 1.0753 / B_resident_block_scarcity: inner 1.0896 · 홀드아웃 1.0893
- **판정**: inconclusive — Inner-CV selects run A (1.0723, against 1.0896 for B). Against H01b (inner 1.0766, holdout 1.0759), run A has holdout 1.0753 and beats the vendor in 3 of 4 folds (f0, f2 and f3; f1 still loses). It picks split_k>=3 on 27/31 hard shapes (H01b: 25/31). Hard gm is 1.0586 (H01b 1.0737) and easy gm is 1.0908 (H01b 1.0779).

Confirm criteria:
- Inner-CV < 1.0766: met (1.0723). The 0.004 margin is within noise.
- Pooled holdout <= 1.0715: missed (1.0753).
- Hungry class (m1_classgap C_hungry, n=31) <= 1.058: missed by 0.0006 (1.0586, against H01b 1.0737 and vendor 1.0695). Its pooled log gap goes from +0.0019 to -0.0049.
- 1-CTA picks on hungry shapes <= 3/31: met (1/31). My recount for H01b gives 9/31, not 7, because it counts every pick with max_blocks_per_sm <= 1, which includes a 128x128x64 st3 pick and a 128x128x32 st6 pick. The last one left in A is 256x4096x4096 at 128x128x32 st6.
- Fold-0 holdout <= 1.06: missed. Fold 0 got worse: 1.0893, against H01b's 1.0812.

Refute criteria:
- 1-CTA picks >= 6/31: not met (1/31).
- Hungry class >= 1.070: not met (1.0586).
- Fitted weight <= 0 in >= 3 folds: not met. A's weight is positive in 4/4 folds, and it stays large relative to the base spread: about 0.7, 3.0, 5.4 and 0.65 base-spread units in f0 to f3.

The verdict is therefore inconclusive. The mechanism is supported, but the headline number does not move.

What happened to the targeted shapes. Inside the gate the residency term does what C4 predicts. On fold 0 the 128x256 sk3 picks become 128x128x32 sk4:
- 1024, 1500, 1536, 2048 and 2304 x4096x4096: from 1.10-1.15 to 1.00-1.04.
- 512x4096x4096: from 1.182 to 1.000.
- f3 1024x4096x2048 and 2048^3: from about 1.10 to about 1.00.

Why the pooled number does not improve. Two side effects cancel the gain:
1. The refit moved shared weights outside the gate. On fold 0, 128x4096x4096 (outside P, log_flops 32 < 33.21) went from 1.080 to 1.672. That one shape adds about +0.0067 to the pooled log gap. Fold-0 log gap per 65: gated shapes -0.0042, non-gated +0.0062.
2. With the smaller 128x128 tile, the split-K quadratic now over-splits by one step:
   - 4096^3: sk4, 1.109 to 1.344 (best is sk3).
   - 3000x4096x4096: 1.092 to 1.219.
   - 768x4096x4096: 1.000 to 1.068.
   - f1 K=11008 shapes: sk8, for example 4096x4096x11008 1.080 to 1.134 and 128x4096x11008 1.114 to 1.215.
   The mid-easy class goes from 1.151 to 1.192, a +0.0069 log gap.

So on a6000, residency (C4) and split count (C1) are coupled. H01b got the split count roughly right partly because it was using 1-CTA 128x256 tiles, which have half as many CTAs. Fixing the tile exposes a split-count error that was hidden before.

Train-only diagnostic, with the base frozen: gated train shapes where the base picks a 1-CTA tile while the optimum is 2-CTA number 13 in f0, 9 in f1, 8 in f2 and 0 in f3. With the base frozen, the indicator lowered train regret in f1 (1.0714 to 1.0634) and f2 (1.0642 to 1.0477) and did nothing in f3. In f0 it raised train regret (1.0810 to 1.1171), because the 1-CTA 128x128x64 picks escaped to 2-CTA 128x64x64 st2 (1.37-2.19). f0 therefore took the +1 prior, and CMA kept the weight positive (+245).

Run B fails as the physics predicts. 1/max_blocks_per_sm also rewards 4-CTA 64x64 tiles; its +k starting point sends every gated train shape to 64x64x32 (train 1.34-1.42). CMA then disabled or reversed the term:
- f0: -787, now favouring 1 CTA.
- f2: 1.8e6 down to 1.5e4, about 0.8% of a base spread.
- f3: 21 down to 0.54.
The resulting 1-CTA hungry picks are 11/31 and hungry gm is 1.090.

C4 is binding inside P only when stated as an indicator for the 1-CTA class. A monotone residency axis cannot express it.
- **상수의 출처**: Gates: copied verbatim from H01b's split-K `if` in each fold. H01b derived them from each fold's train shapes: f0 log_flops>=33.2131 and owps<234057; f1 and f3 log_flops>=32.5 and owps<234057; f2 owps<249661 and roofline_ratio>=1.32805. Feature: /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/r2_H18/feat_src1cta.jsonl is the single_resident_cta line copied byte-for-byte from feat_pershape2.jsonl (checked with diff). It reads only cfg.max_blocks_per_sm. w0[n] = sign_f x k_f, computed by prep.py on fold-f TRAIN shapes inside the gate only (_splits(t, fold=f, k=4, design='nkband').train.shapes; gated train n = 23/28/26/27). k_f = median over gated train shapes of the IQR/1.349 within-shape spread of the H01b base score (at H01b w0), divided by the median within-shape std of the term. Base spreads for f0..f3 are 156.4, 21.34, 4.526e5 and 5.256. Term std is 0.4454 for single_resident_cta and 0.2494 for resident_block_scarcity. So k_A = 351.07, 47.92, 1.016e6, 11.80 and k_B = 626.9, 85.58, 1.815e6, 21.08. Sign comes from frozen-base train regret (gm over all train shapes). A: f0 both signs worse than 0 (+k 1.1171, -k 1.0948, 0 1.0810), so +1 prior; f1 +1 (1.0634 < 1.0714); f2 +1 (1.0477 < 1.0642); f3 +1 (+k equals 0 at 1.0651, -k 1.0959). B: both signs worse than 0 in every fold, so +1 prior. No holdout shape or result was used for any constant or for the selection. Holdout was read only afterwards, for reporting: post.py and diff.py recompute picks from the fitted w and assert they match the official regrets. Files: prep.py/prep.log/prep.json, diag2.py/diag2.log, build.py, H18A_1cta.json, H18B_rbs.json, post.py/post.log/post.json, diff.py/diff.log, all in /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/r2_H18/.

### 6. H19 — Over-split guard outside the split-K gate fails and makes things worse; the graded large-grid ramp works on its own class

- **시험한 원인**: This extends C1. The claim: on the a6000, over-splitting is costly outside band P too, and the rule has no penalty for very short K-slices there. It also refines C3: a graded large-grid ramp, instead of H01b's on/off owps>2e5 gate, should rescue the 8192-row swizzle-trap grids without hurting the 2048-row grids.
- **바꾼 것**: Both runs start from H01b per fold. Run A adds an `else:` to each fold's split-K gate. It charges f.split_overhead_fraction, which is (stages+1)/iterations-per-split, copied verbatim into a one-line jsonl. Its start weight is +k_f, set from train shapes outside the gate. Run B is Run A plus one change: the on/off large-grid block becomes `if p.roofline_ratio >= 1:` with its terms multiplied by max(0, log2(owps/c_f)). c_f is the train owps threshold where the 1-CTA swizzle-1 config starts beating the 2-CTA one. The ramp weights are H01b's divided by the median train ramp.
- **돌린 것**: A_guard (H01b + split_overhead_fraction guard outside gate, w0=+k_f): inner 1.0933 · 홀드아웃 1.1127 / B_guard_ramp (A + graded large-grid ramp max(0,log2(owps/c_f))): inner 1.0884 · 홀드아웃 1.1052
- **판정**: refuted — Inner-CV selects Run B: 1.0884, against 1.0933 for A. Both runs are clearly worse than H01b (inner 1.0766, holdout 1.0759) and worse than the vendor (1.0815). B's holdout is 1.1052 and it beats the vendor in 1 of 4 folds. A's holdout is 1.1127, beating the vendor in 2 of 4. The confirm conditions fail: the selected run needed inner < 1.0766 and holdout <= 1.0715.

**Guard part (Run A): refuted by its own criterion.**
- Mid-easy is 1.305 in A and 1.209 in B, both at or above 1.14 (H01b 1.151).
- Fold 1 is 1.253 in A and 1.134 in B, both at or above 1.13.
- The fitted guard weight stayed positive in all 4 folds, so the first refute clause did not trigger:
  - A: f0 126→56.6, f1 163→114.7, f2 280439→492, f3 38.6→213.9.
  - B: 49.6, 69.2, 189.8, 66.9.
  - In f2 the fitted 492 is about 0.2% of the base's robust within-shape spread (7.6e4), so it is effectively zero.
- Other Run A checks:
  - M<=32 is 1.014, which passes (<= 1.025).
  - No holdout pick outside the gate is at sk>=8 in A. B still picks sk8 on 512^3 (1.818).

**Why the guard fails, part 1: over-splitting outside P is barely visible in train.** This is the train diagnostic from before the runs, done with the frozen H01b, using both its w0 and its full-train fitted w.
- Train shapes outside the gate where H01b picks sk>=4: none in folds 0, 1 or 2.
- In fold 3 there are 4 such shapes, all at sk4 where the best is sk1: 1x/8x/32x4096x11008 at 1.04-1.05, and 128x4096x4096 at 1.14.
- The 512^3 sk8 disaster (1.818) comes from the fold-1 fit. That shape is not in fold-1 train. In folds 0 and 2, where it is in train, the base picks sk1.

**Why the guard fails, part 2: the axis is not a split-K-only cost.** At sk=1 on short K, (stages+1)/iterations acts as a stage-count penalty, and that moves tiles.
- In fold-1 holdout, Run A does fix 512^3 (1.818→1.091, now sk1).
- But it blows up 1024x4096x128 (1.385→5.923, picks 256x256x32 at sk1) and 128x4096x256 (1.091→2.0).
- The frozen guard at +k_f raised train gm in every fold: f0 1.081→1.182, f1 1.071→1.076, f2 1.064→1.228, f3 1.065→1.068.

**Why the guard fails, part 3: the large start weight hurts the CMA fit.** Full-train fits end worse than H01b's:
- A f2: train 1.0966 vs 1.0632.
- B f0: train 1.0983 vs 1.0675.

That is why hungry shapes inside the gate regress even though the guard is off there:
- f2 256x11008x4096, 512x11008x4096 and 1024x4100x4096 go from sk3 to sk1, at 1.12-1.23.
- f0 384x4096x4096 goes to 1.507 in B.

Hungry class: A 1.095, B 1.120, against H01b 1.074. sk>=3 picked on hard shapes: 22/31 in A, 13/31 in B.

**Ramp part (Run B): not refuted, and it meets most of its class criteria.**
- Each fold's train data has a separating threshold. Misclassified shapes: f0 1/11, f1 0/13, f2 2/11, f3 2/10.
- c_f is 345940, 463052, 489234 and 463052.
- Large-grid class is 1.042, down from H01b's 1.060, which passes (<= 1.045). Its log gap against the vendor goes from +0.0037 to +0.0013.
- Both 2048-row shapes return to L1's 1.013 (H01b had 1.101 and 1.087), which passes (<= 1.03). Their ramp is 0 because owps 2.7e5-3.0e5 is below c_f.
- 8192x11008x4096 (1.033), 8192x12288x4096 (1.030) and 8192^3 (1.037) stay rescued.
- 8192x4096x4096 (1.078) and 8192x4096x11008 (1.070) fail the <= 1.05 bar. They are unchanged from H01b. Their owps is 3.995e5, near or below c_f, and they lose because they pick 1-CTA 128x256 sk1 where 128x128 sk2 is best. The ramp does not change them.

**What this means for the causes.** C1 does not extend to "over-splitting outside P" in any way train can see. What remains is a holdout-only 512^3 artefact of fold 1, and a generic (stages+1)/iters penalty costs more than it saves. C3's graded ramp is a real improvement over the on/off gate, but here it is swamped by the guard's damage.

A post-hoc hint only, computed from holdout values and not a result: swapping B's large-grid regrets into H01b would give about 1.073 pooled.
- **상수의 출처**: - **Gates and base code:** copied from H01b, whose gates were derived from train in H01.
- **split_overhead_fraction:** line 1 of an/pershape/cand/feat_pershape.jsonl, copied verbatim (diff shows it identical). It reads only p.K, cfg.tile_k, cfg.split_k and cfg.stages.
- **Guard weight k_f** (it/r1_H01/scale.py robust procedure, fold-f TRAIN shapes outside that fold's split-K gate only): k_f = median of (IQR/1.349 of the H01b base score, using H01b code and H01b w0) divided by the median within-shape std of the term. The sign is + by physics.
  - f0: 25 shapes, 34.25 / 0.2713 = 126.233
  - f1: 21 shapes, 44.31 / 0.2713 = 163.301
  - f2: 23 shapes, 7.61e4 / 0.2713 = 280439
  - f3: 22 shapes, 10.48 / 0.2713 = 38.6306
- **Ramp threshold c_f** (fold-f TRAIN shapes with roofline_ratio >= 1 and owps > 1e5):
  - Labelling: t1 = min train time over candidates with max_blocks_per_sm==1 and ext_swizzle_n==1; t2 = the same with max_blocks_per_sm==2. A shape is trap iff t1 < t2.
  - Threshold rule: the 1-D owps threshold with the fewest misclassified shapes, ties broken toward the larger one. c_f is the geometric mean of the two adjacent owps values.
  - f0: sqrt(299593 x 399458) = 345940, 1/11 misclassified
  - f1: sqrt(399458 x 536771) = 463052, 0/13
  - f2: sqrt(399458 x 599186) = 489234, 2/11 (tie broken toward the larger threshold)
  - f3: sqrt(399458 x 536771) = 463052, 2/10 (tie broken toward the larger threshold)
- **Ramp weights:** H01b w0[a,b] divided by the median of max(0, log2(owps/c_f)) over train shapes with roofline_ratio >= 1 and ramp > 0.
  - f0: median 1.000 over 6 shapes, weights unchanged at -2.393, -2.665
  - f1: median 0.787 over 5 shapes, 103.31, -115.05
  - f2: median 0.708 over 3 shapes, -31134, -34671
  - f3: median 0.713 over 2 shapes, -27.54, -30.66
- **Holdout:** no holdout shape or holdout result was used for any constant.
- **Diagnostic:** also used H01b's full-train fitted w, which was fit on train only.

### 7. H20 — H01b recipe on the s1 base with formula constants: it transfers (Run A confirmed), but the same formulas do not reproduce H01b on s0 (Run B refuted)

- **시험한 원인**: Is the C1+C3 fix (a gated split-K bowl plus a large-grid tile-choice branch) general, or does it only work on the s0 base with the per-fold b sweep from s25 (b = 100/1/3e5/1, centre 5.5; in the file, f1's centre was actually 6.0)? This merges H05 and H11.
- **바꾼 것**: Run A adds the two H01b blocks to the s1 base. The first is a bowl-shaped split-K term inside each fold's train-chosen gate. The second is the large-grid branch on resident_block_scarcity and register_file_pressure. Their starting weights come from formulas on train shapes, with no sweep. Run B keeps H01b exactly as it is and replaces only the swept starting weights of its split-K bowl with the same formulas computed on the s0 base. The bowl's centre is the median log2(K-iterations per split) of the train-optimal configs. Its strength matches the bowl's spread within each shape to the base score's spread within each shape.
- **돌린 것**: A_s1_formula: inner 1.0771 · 홀드아웃 1.0710 / B_H01b_formula_quad: inner 1.0931 · 홀드아웃 1.0902
- **판정**: inconclusive — The two parts point in opposite directions, so the combined verdict is inconclusive. On its own, Run A is confirmed and Run B is refuted.

**Run A (s1 base, formula constants) meets all four confirm conditions**
- Pooled holdout is 1.0710, against the s1 baseline's 1.1023 (0.031 lower; needed 0.015 lower and at most 1.085).
- Inner-CV is 1.0771, against 1.0976 (0.021 lower; needed 0.01).
- The hungry class (sk>=3, n=31) scores 1.064; it needed to be at most 1.08.
- sk>=3 is picked on 27/31 hard shapes; it needed at least 22.
- It beats the vendor in 3 of 4 folds, and 1.0710 is 0.0105 below the vendor's 1.0815. That margin is about the noise level.

**Per-class log gap against the vendor (m1_classgap classes), s1 → A**
| Class | s1 | A | H01b |
|---|---|---|---|
| hungry | 1.132 (+0.0271) | 1.064 (-0.0025) | 1.074 |
| large-grid | 1.121 (+0.0114) | 1.087 (+0.0072) | 1.060 |
| M<=32 | 1.018 | 1.018 (unchanged) | – |
| mid-easy | 1.101 (-0.0002) | 1.129 (+0.0048) | – |

- Hungry is the best number recorded for that class.
- Large-grid improves, but less than H01b.
- The mid-easy loss comes almost entirely from f1 512x512x512, which went from 1.091 to 1.656. That shape is outside both gates, so the loss comes from refitting the shared s1 weights, not from the new blocks.
- f2 8192x11008x4096 fell into the swizzle trap, going from 1.030 to 1.328. That keeps large-grid above H01b.
- Fold 1 still loses to the vendor, 1.114 against 1.089. It does improve on H01b's 1.137.
- All four fitted split-K terms stayed bowl-shaped. Centres are 4.79 / 5.04 / 5.38 / 4.95, and the squared-term weights grew from w0 (715 / 6.85 / 41.3 / 589) to 1743 / 6.95 / 31.2 / 1060.

So the C1+C3 fix is not specific to the s0 base. On a different evolved base it repairs exactly the two weak classes, hungry and large-grid. That supports C1 (the interior optimum is what is needed) and C3.

**New best? No, by the stated rule.** A's inner-CV of 1.07711 is 0.0005 above H01b's 1.0766, which is a tie. Only A's holdout is lower (1.0710 against 1.0759), and holdout is not the selection metric. H01b and A are tied on inner-CV. A is the first rule to reach H01's 1.0715 holdout goal.

**Run B (H01b with formula split-K w0 on s0) meets the refute condition**
- Inner-CV is 1.0931, 0.0165 worse than H01b.
- Holdout is 1.0902, 0.0143 worse.
- sk>=3 on only 13/31 hard shapes, hungry 1.102, 2 of 4 folds.

The failure is confined to f0 and f2, where the formula strength is far below the swept one: b = 2.20 against 100 (45x smaller) and 10504 against 3e5 (29x smaller). In f1 and f3 the formula is 12.5 against 1 and 3.1 against 1, and holdout stays close to H01b (f1 1.143 against 1.137, f3 1.042 against 1.039).

The fitted weights show why. In f0 and f2 the refit flipped the squared weight negative, to -8.36 and -2686. The term became a hill instead of a bowl, and the picks fell back to sk 1-2: in f0, 13 of 17 holdout shapes pick sk=1; in f2, 15 of 16 do. This reproduces the H06 control, where a term without a bowl loses the split-K gain. A train-only check with the s0 weights frozen agrees. At the formula b, the f0 gated train shapes still pick sk=1 on 18 of 23, while the swept b picks sk>=3 on 21 of 23.

So H01b's gain on s0 depends on the per-fold b sweep. The IQR-matching formula works on s1, where the base score spreads widely within each shape (median IQR 1155 / 11 / 66 / 948). It underestimates the needed strength on s0 f0 and f2, where the base's spread within each shape is small (3.55 in f0) or dominated by a few huge weights.

**What this means for the causes.** C1 and C3 are general across bases. C5, the weight-scale problem, is still live: a sweep-free w0 has to measure the base's anti-split margin, not its overall spread.

**Constants.**
- c_f = 5.415 in every fold and for both bases, from 23 / 28 / 26 / 27 gated train shapes.
- Run A: b_f = 714.83 / 6.848 / 41.263 / 589.48.
- Run B: b_f = 2.198 / 12.514 / 10504 / 3.106.
- **상수의 출처**: Everything comes from each fold's TRAIN shapes (`_splits(t, fold=f, k=4, design='nkband').train.shapes`), computed in derive.py; no holdout shape or holdout result was read.

**Split-K gate:** taken verbatim from an/logs/fold_gates.json, which s19 chose on train.

**Centre c_f:** the median, over gated train shapes, of L = log2(mainloop_iteration_count / (1 + serial_reduction_dependency * mainloop_iteration_count)) at that shape's argmin-measured-time config. That config comes from train times only. The result is 5.415 = log2(128/3) in all folds.

**Strength b_f:** the median over the same shapes of IQR/1.349 of the base score, divided by the median of IQR/1.349 of (L - c_f)^2.
- The base score is s1 code at s1 w0 for Run A, and c2_evolved_s0 code at s0 w0 for Run B.
- The bowl's IQR/1.349 medians are about 1.61.
- The base's IQR/1.349 medians are 1155 / 11.02 / 66.37 / 948.2 for s1, and 3.55 / 20.13 / 16900 / 4.996 for s0.
- The starting weights of the bowl are w0 = (-2 c_f b_f, b_f).

**Large-grid block (Run A only):**
- Gate: roofline_ratio >= 1 and output_work_per_sm > 200000, copied from r09/H01b. On train it covers 8 / 8 / 6 / 5 shapes, none of which are inside the split-K gate. So the base score here equals s1's own score, whether or not the split-K block is added.
- Scale: the median IQR/1.349 of the base score, divided by the median std of each term, as in r1_H01/scale.py. This gives [490.44, 546.15], [44.00, 49.00], [268.68, 299.20] and [3542.0, 3944.4] per fold.
- Sign: the r1_H01 procedure. All 9 sign combinations are scored on train at the std-ratio scale, and the best of the four with no zero sign is taken, with itertools order breaking ties. I checked that this reproduces H01b's signs in all 4 folds. It gives (-1,-1) in every fold. The sign on resident_block_scarcity is tied with 0 and +1 in f0, f1 and f3, so the tie order decides it there.

**Run B:** keeps H01b's code, extra_features and all other w0 values unchanged. Only w[28:30], w[34:36], w[41:43] and w[31:33] were replaced: f0 (-1100, 100) → (-23.80, 2.198), f1 (-12, 1) → (-135.53, 12.514), f2 (-3.3e6, 3e5) → (-113758, 10504), f3 (-11, 1) → (-33.64, 3.106).

**Held fixed in Run A:** s1's code, w0 (lengths 17 / 29 / 35 / 40) and extra_features runs/c2-a6000-f{f}-s1/features.jsonl are unchanged.

### 8. H07b — Control for C2: the split-K quadratic behind the textbook gate owps<4096 loses H01b's hungry-shape gains (confirmed)

- **시험한 원인**: CONTROL for C2 (gate location). On the a6000, split-K need sits in band P (log_flops >= ~32.5 AND owps < ~2.34e5), not behind the loop's usual 'small output per SM' gate. Moving H01b's interior-optimum split-K term behind owps<4096 should lose the hungry-class gains, and may disturb M<=32 and tiny shapes.
- **바꾼 것**: I took the current best rule, H01b_robust, and in each of the 4 folds changed only the `if` that switches on the split-K 'sweet spot' term. It now reads `if p.output_work_per_sm < 4096:` instead of the band-P test. The two quadratic lines, every w0 value, the large-grid block and extra_features are all unchanged, so this is the main run. As a second run I kept the same code and re-derived only the quadratic's start weights for the new gate, from train shapes only (w0 = [-2bc, b], with centre c and scale b chosen by a train-regret sweep). This gives the moved gate its fairest chance.
- **돌린 것**: H07b_main (gate -> owps<4096, all w0 unchanged): inner 1.1033 · 홀드아웃 1.0989 / H07b_retuned (same gate; quadratic w0 re-derived on train inside owps<4096): inner 1.1038 · 홀드아웃 1.0947
- **판정**: confirmed — Inner-CV selects H07b_main: 1.1033 against 1.1038 for the retuned run, which is a tie. Every confirm condition holds for the selected run, and no refute condition does.

**Confirm conditions (selected run)**
- **Hungry class:** 1.120 (needs >= 1.10). H01b had 1.074 and the vendor has 1.069.
- **Split-K on hard shapes:** sk>=3 is picked on 13/31 hard shapes (needs <= 16). H01b picked it on 25/31.
- **Pooled holdout:** 1.0989 (needs >= 1.095). H01b had 1.0759 and the vendor has 1.0815.
- **Folds beating vendor:** 1 of 4 (needs <= 1). Only f3 beats it.
- **Large-grid class:** 1.054, against H01b's 1.060. It is unchanged, so the loss sits in the hungry class.

**Refute conditions:** pooled holdout <= 1.085 or hungry <= 1.085. Neither happened: pooled is 1.099 and hungry is 1.120.

**Size of the effect.** Moving the gate costs +0.027 inner-CV (1.0766 to 1.1033) and +0.023 pooled holdout. Both are well above the ~0.01 noise.

**Where the loss is.** In pooled log gap against the vendor, per 65 shapes, going from H01b to H07b_main:

| Class | H01b | H07b_main |
|---|---|---|
| hungry | +0.0019 | +0.0222 |
| large-grid | +0.0037 | +0.0029 |
| mid-easy | +0.0088 | +0.0075 |
| M<=32 | -0.0195 | -0.0166 |

The worst hungry-shape losses:
- 384x4096x4096: 1.074 to 1.507
- 3000x4096x4096: 1.092 to 1.346
- 256x11008x4096: 1.028 to 1.226
- 768x4096x4096: 1.000 to 1.216

**Why this happens.** Over all 65 shapes, the owps<4096 gate opens for 0 of the 31 hungry shapes and 11 of the 34 others. On train it opens for 0 hungry shapes in every fold: 8, 7, 9 and 9 train shapes, all M<=32 or 512^3, all with optimum split_k 1-2. So the interior-optimum term never sees a shape that needs split_k 3-4. The refit then turns it into a one-direction term: the fitted parabola vertices are at L = 162, -222, -62 and 15, far outside the data.

**Fold detail.** The split-K picks collapse only where the base depends on the P-gated quadratic:
- f0: 9/11 hungry holdout shapes picked sk>=3 in H01b, 0/11 now.
- f2: 4/5 before, 0/5 now.
- f1: 7/7 before, 6/7 now.
- f3: 5/8 before, 7/8 now.

In f1 and f3 the evolved base code has its own split-K terms (split_k_partition_count x wave terms, and a wave_geometry_factor), so those folds do not need gate P.

**The predicted side effect on M<=32 and tiny shapes is real but small.** The M<=32 class goes from 1.016 to 1.033, still far below the vendor's 1.130. Examples: f0 8x4096x4096 goes from 1.034 to 1.086, and f1 1x4096x11008 moves from sk2 to sk1 and from 1.008 to 1.051. 512^3 is unchanged at 1.818.

**The retuned run gives the same picture.** Its (b, c) came from train inside the new gate. Hungry is 1.112, sk>=3 is 13/31, 1 fold beats vendor, and pooled holdout is 1.0947. That misses the >= 1.095 confirm line by 0.0003, which is inside noise. Re-tuning the start scale cannot bring the split-K picks back, because the gate holds no hungry shapes.

**Meaning for the cause.** Where the split-K term is switched on matters, not just how it is scored. The textbook 'small output per SM' gate points at the wrong shapes on the a6000. Band P (large flops, moderate owps) is where the interior-optimum term earns H01b's gain, so C2 is a real, separate cause. That holds on the c2-s0-style base (f0) and on the f2 base. The f1 and f3 bases carry split-K information elsewhere.
- **상수의 출처**: The main run adds one new constant, 4096 in `p.output_work_per_sm < 4096`. It is the textbook gate named in the hypothesis, and the same literal already appears in the evolved c2 code (f2 and f3 bases). It was not fitted to anything. Everything else is copied unchanged from H01b_robust.json: the quadratic lines, all w0, the large-grid gate (roofline_ratio >= 1, owps > 200000), and extra_features.

The retuned run changes only the two quadratic w0 entries, to [-2bc, b]. For each fold, (b, c) came from sweep_gate.py, which reads only `_splits(t, fold=f, k=4, design='nkband').train.shapes`. It freezes the other H01b start weights and adds b*(L - c)^2 on the train shapes with owps < 4096, where L = log2(mainloop_iteration_count / (1 + serial_reduction_dependency * mainloop_iteration_count)). The grid was c in {3, 3.5, ..., 7} and b = ±{0.01, ..., 100} x |H01b's b|, and the pick is the lowest train gm. Ties (train gm equal to 1e-6) go to the b closest in log to H01b's b, then to the c closest to H01b's c.

| Fold | b | c | w0 before | w0 after |
|---|---|---|---|---|
| f0 | 3 | 7 | [-1100, 100] | [-42, 3] |
| f1 | 1 | 6 | [-12, 1] | [-12, 1] (unchanged, all ties) |
| f2 | 3e5 | 6.5 | [-3.3e6, 3e5] | [-3.9e6, 3e5] |
| f3 | 1 | 7 | [-11, 1] | [-14, 1] |

Holdout shapes and holdout results were never used in building either rule. The class breakdown, and the 0/31 and 11/34 gate counts over all 65 shapes, were computed after the runs, for reporting only.

### 9. H21 — Graded large-grid ramp without the guard: fixes the big class on the H18A base, but the refit breaks fold 0's split-K on both bases and wrecks one s1 shape (inconclusive)

- **시험한 원인**: C3 refined. The swizzle-trap cost grows with output work per SM, so a train-thresholded graded ramp max(0, log2(owps/c_f)) should rescue the 8192-row grids and leave the 2048-row grids alone. Without H19's guard it should improve the big class and not touch the hungry or mid-easy classes. Two bases (H18A, H20A) test whether this depends on the base.
- **바꾼 것**: In both co-best rules, I replaced the on/off large-grid branch (compute-bound and owps > 2e5) with a graded ramp. The same two terms, resident_block_scarcity and register_file_pressure, are now multiplied by max(0, log2(owps / c_f)). The ramp is zero below each fold's train threshold c_f and grows above it. Their start weights were divided by the fold's median train ramp value, so a typical big shape starts from the same strength as before. Run A uses H18A as the base and Run B uses H20A. Nothing else in the rules changed; a diff confirms 3 code lines and 2 w0 entries per fold.
- **돌린 것**: A_H18A_ramp: inner 1.0889 · 홀드아웃 1.0764 / B_H20A_ramp: inner 1.0848 · 홀드아웃 1.1202
- **판정**: inconclusive — **Selection.** Inner-CV selects Run B: 1.0848 against 1.0889 for A. The gap is 0.004, inside noise, and B has the worse holdout: 1.1202 against 1.0764 for A. Both runs are worse than their bases on inner-CV: A 1.0889 against H18A's 1.0723, and B 1.0848 against H20A's 1.0771. Neither beats the vendor's pooled 1.0815. A beats the vendor in 2 of 4 folds, B in 1 of 4.

**Confirm conditions**
- Selected run inner < 1.0723: missed (1.0848).
- Selected run holdout < 1.0710: missed (1.1202).
- At least 3 folds beating the vendor: missed (1 of 4).
- Big class <= 1.045 in both runs: met in A (1.041, from 1.060), missed in B (1.462, from 1.087).
- Both 2048-row shapes <= 1.03 in A: met (1.013 and 1.013).
- Hungry class moves by at most +0.01: missed in both runs (A 1.059 to 1.079, B 1.064 to 1.079).

**Refute conditions**
- "Big class not at least 0.01 below its base in both runs": not met, because A improves it by 0.019.
- Pooled holdout worse than its base by >= 0.01: B meets it (+0.049), A does not (+0.001).

Read literally, that is a mixed result, so the verdict is inconclusive. Run B on its own would count as refuted.

**Per-class holdout (r3_classgap classes)**
| Class (n, vendor) | H18A | H21A | H20A | H21B |
|---|---|---|---|---|
| M<=32 (12, 1.130) | 1.012 | 1.015 | 1.018 | 1.014 |
| big (9, 1.032) | 1.060 | 1.041 | 1.087 | 1.462 |
| hungry (31, 1.069) | 1.059 | 1.079 | 1.064 | 1.079 |
| mid-easy (13, 1.102) | 1.192 | 1.156 | 1.129 | 1.117 |

**Named shapes: regret [pick]**
| Shape (fold) | H18A | H21A | H20A | H21B |
|---|---|---|---|---|
| 2048x11008x4096 (f2) | 1.101 | 1.013 [128x128x32 sk1] | 1.092 | 1.013 [128x128x32 sk1] |
| 2048x12288x4096 (f3) | 1.087 | 1.013 | 1.087 | 1.013 |
| 8192x11008x4096 (f2) | 1.033 | 1.033 [128x256x32 sk1] | 1.328 | 17.288 [256x256x32 sk1] |
| 8192x12288x4096 (f3) | 1.030 | 1.030 [256x128x32] | 1.030 | 1.030 |
| 8192^3 (f3) | 1.037 | 1.037 [256x128x32] | 1.037 | 1.037 |
| 8192x4096x4096 (f0) | 1.078 | 1.062 [128x256x32 sk1] | 1.078 | 1.072 [256x128x32 sk1] |

**What this says about C3**
- The grading itself works and is not a side effect of H19's guard. Run A has no guard and reproduces H19B's big-class result: 1.041 against 1.042, with the same picks. Both 2048-row shapes return from about 1.09-1.10 to 1.013, because their ramp is 0 (owps 2.7e5-3.0e5 is below c_f). The 8192-row rescues are kept.
- It is base-dependent. On the s1 base, fold 2, the start weight on register_file_pressure (-422.9, i.e. H20A's -299 / 0.708) is multiplied by a ramp of 1.13. That sends 8192x11008x4096 to the 1-CTA 256x256x32 tile, with regret 17.3, which alone adds +0.040 to the pooled log gap.
- The fold-2 fit never moved those two weights. Train gives it nothing to push against: the only fold-2 train shapes with ramp > 0 are 8192x12288x4096, 8192^3 and 4096x12288x4096, and none of them picks 256x256 there.
- This is a different failure from the partial-result case the hypothesis anticipated (a 128x128x64 pick), but the conclusion is the same: C3's tile route depends on the base.

**Why hungry and inner-CV got worse: a refit path failure in fold 0, not the ramp's physics**
- In fold 0, all 11 hungry holdout shapes collapse to split_k=1 in both runs, with identical regrets: 384x4096x4096 1.507, 1024x4096x4096 1.175, 512x4096x4096 1.133, 768x4096x4096 1.216. The ramp is exactly 0 on those shapes, since their owps is below 2.4e5 and c_f is 3.46e5.
- Train-only diagnostic (diag.log):
  - With the base's fitted weights transplanted into the ramp code (a,b / median_ramp), fold-0 train gm improves: A 1.0493 to 1.0458, B 1.0426 to 1.0391. Hungry train shapes keep split_k>=3 on 18 of 20.
  - The CMA refit of the new code ends much worse: train 1.0935 (A) and 1.0832 (B), with 0 of 20 hungry train shapes at split_k>=3.
  - In A the bowl flips into a hill: the squared weight goes from +220.7 to -81.9.
  - The fit starts at w0 (train 1.117), so a small change in the objective sends the deterministic CMA down a worse path.
- The same fold-0 collapse appears in H19B, with identical regrets on 512, 1000, 384 and 768x4096x4096. So H19's attribution of its fold-0 damage to the guard is at least partly wrong: it is refit fragility (C5-type) triggered by the ramp's change to the code.
- In Run B, fold 1 also loses split-K picks under the refit (hungry train split_k>=3 falls from 18/24 to 8/24).
- Mid-easy improves in both runs. The biggest single move is A fold 0 128x4096x4096, 1.672 to 1.094, which is also a refit effect.
- **상수의 출처**: - **c_f and median_ramp** come from it/r2_H19/prep.json, computed on each fold's TRAIN shapes only.
  - c_f is the geometric midpoint between the highest train owps where 2-CTA swizzle-1 is not a trap and the lowest owps where it is. For fold 0 that is sqrt(299593 x 399458) = 345940.4. The values are 345940.4 / 463052.1 / 489233.6 / 463052.1.
  - median_ramp is the median of max(0, log2(owps/c_f)) over the fold's train large-grid shapes with ramp > 0: 1.0 / 0.78687 / 0.70752 / 0.71313.
- **Run A w0[a,b]** = H18A's w0 (itself H01b's r09-derived w0) / median_ramp:
  - f0 -2.393455, -2.665348
  - f1 103.3148, -115.0512
  - f2 -31133.98, -34670.76
  - f3 -27.53506, -30.66301
  These are identical to H19B's ramp w0.
- **Run B w0[a,b]** = H20A's large-grid w0 (from r2_H20/derive.py) / the same median_ramp:
  - f0 -490.4399, -546.1533
  - f1 -55.91732, -62.26946
  - f2 -379.7513, -422.8906
  - f3 -4966.821, -5531.046
- **Indices:** A (30,31) / (36,37) / (43,44) / (33,34); B (19,20) / (31,32) / (37,38) / (42,43).
- All other constants (the P gate, the bowl, the 1-CTA indicator, the base terms) are copied unchanged from the base files.
- No holdout shape or result was used to set any constant. The diagnostics score train shapes only.

### 10. H22 — H20's formula-constant recipe on the s3 and s2 bases: big holdout gains and bowls that stay bowls, but inner-CV improves by under 0.01 and the hungry class stays at 1.10 (inconclusive)

- **시험한 원인**: C1+C3 generality: the gated split-K bowl in band P (C1/C2) plus the large-grid tile-choice branch (C3). Built with H20A's sweep-free train-formula constants, it should repair the hungry and large-grid classes on the s3 and s2 bases as it did on s1. The test also checks whether the IQR-scaled bowl flips into a hill (the C5 failure mode seen in H20B).
- **바꾼 것**: I copied H20's derive.py and changed only the base rule it reads: c2_evolved_s3 for Run A and c2_evolved_s2 for Run B, each with its own runs/c2-a6000-f{k}-s{3|2}/features.jsonl. To each fold's code I appended exactly H20A's two blocks. The first sits inside the fold's band-P gate and is a bowl-shaped term in L = log2(K-loop iterations per serial split), with weights w[n] and w[n+1]. The second sits inside `roofline_ratio >= 1 and owps > 200000` and adds resident_block_scarcity·w[n+2] + register_file_pressure·w[n+3]. All new constants come from each fold's TRAIN shapes through H20's formulas, with no sweep, no ramp and no 1-CTA indicator. Both rules passed --check-only on all 4 folds, and I used exactly one official run for each.
- **돌린 것**: A_s3_formula (s3 base + H20A blocks): inner 1.0806 · 홀드아웃 1.0826 / B_s2_formula (s2 base + H20A blocks): inner 1.0927 · 홀드아웃 1.0986
- **판정**: inconclusive — Inner-CV selects Run A: 1.0806, against 1.0927 for B. A's pooled holdout is 1.0826, which ties the vendor's 1.0815 within noise, and it beats the vendor in 2 of 4 folds (f0, and f2 by 0.001). It is not a new best: H18A has inner 1.0723, H01b 1.0766 and H20A 1.0771.

**Confirm criteria: all four fail**
- Inner-CV improvement of at least 0.01: fails. A improves by 0.0063 (1.0868 to 1.0806) and B by 0.0086 (1.1013 to 1.0927).
- Holdout improvement of at least 0.015: met by a wide margin. A improves by 0.043 (1.1258 to 1.0826) and B by 0.047 (1.1460 to 1.0986). But holdout is not the selection metric.
- Hungry class at most 1.08: fails. The class is the 31 shapes whose optimum has sk≥3, the same set as hard_gm. A scores 1.0996 (base 1.160) and B 1.0967 (base 1.155).
- sk≥3 on at least 22/31 hard shapes in both runs: fails. A gets 24/31 (base 10/31) but B only 21/31 (base 5/31).
- Squared weight positive in at least 3/4 folds: met, in fact 8/8.
- New best (inner < 1.0723 and holdout < 1.0710): fails.

**Refute criteria: none triggered**
- Both runs improve inner by at least 0.005.
- No bowl flipped into a hill, so the conjunction with hungry ≥ 1.10 cannot fire. A's hungry class sits at 1.0996, right on that line.

**Fitted bowls**
| Run | Fold | Squared weight, w0 → fit | Vertex L |
|---|---|---|---|
| A | f0 | 14.2 → 38.2 | 5.01 |
| A | f1 | 12.1 → 52.9 | 5.10 |
| A | f2 | 26.1 → 60.5 | 5.68 |
| A | f3 | 4.79 → 30.4 | 5.68 |
| B | f0 | 15592 → 55680 | 5.04 |
| B | f1 | 219 → 23.1 | 5.18 |
| B | f2 | 11.4 → 41.8 | 4.36 |
| B | f3 | 44.4 → 20.7 | 5.03 |

**Logged per-fold scales.** Median within-shape robust spread of the base score (IQR/1.349) over gated train shapes, followed by b_f:
- s3: 22.93 / 19.42 / 42.05 / 7.70, giving b_f 14.19 / 12.07 / 26.14 / 4.79.
- s2: 25190 / 352.9 / 18.40 / 71.36, giving b_f 15592 / 219.4 / 11.44 / 44.36.
- The median over all train shapes is almost the same (s3: 23.9 / 19.2 / 42.1 / 7.8).

The smallest spread (s3 f3, 7.7) did not produce a hill, so the H20B mechanism (IQR too small, CMA flips the bowl) did not recur.

**Per-class results** (m1_classgap classes, pooled log gap against the vendor per 65 shapes)
| Class | s3 | Run A | s2 | Run B | H20A |
|---|---|---|---|---|---|
| Large-grid | 1.146 (+0.0144) | 1.050 (+0.0024) | 1.113 (+0.0103) | 1.052 (+0.0026) | 1.087 |
| Hungry | 1.160 (+0.0385) | 1.100 (+0.0133) | 1.155 (+0.0367) | 1.097 (+0.0120) | 1.064 |
| M≤32 | 1.031 | 1.019 | 1.046 | 1.018 | – |
| Mid-easy | 1.125 | 1.126 | 1.249 | 1.220 | – |

Large-grid ends better than H20A in both runs. The swizzle-trap grids are rescued through 1-CTA 128x256 tiles:
- f3 8192³: 1.436 to 1.033.
- f3 8192x12288x4096: 1.334 to 1.018.
- f1 8192x4096x11008: 1.358 to 1.058.

The fitted resident_block_scarcity weight is negative in 8/8 folds, as in H01. So C3's tile-choice route transfers cleanly.

Hungry picks move to split-K and fix many shapes. In A:
- f0 1500/1536/2048x4096x4096: about 1.24 to 1.00.
- 768x4096x4096: 1.612 to 1.068.
- f1 1024x4096x11008: 1.230 to 1.001.

But one-step over-splits and wrong tiles remain:
- f1 4096x4096x11008 at sk12 (best sk4): 1.468.
- f0 3000x4096x4096 at sk4 64x128: 1.412.
- f0 4096³ at sk4 (best sk3): 1.344, the same over-split H18 exposed.
- f3 128x12288x4096 at 64x64 sk2: 1.503.
- In f2, hard shapes keep 1-CTA 256x128 at sk 1-3 (fold hard gm 1.163).

**Why inner-CV gains are small despite the big holdout gains.** The new rule contains the base as a special case, since new weights of zero give back the base. Even so, the full-train fit ended worse than the base's own fit in the folds where the base already splits:
- A f2: train 1.0892 against 1.0817.
- A f3: 1.0796 against 1.0677.
- B f3: 1.0885 against 1.0767.

Those are exactly the folds where inner-CV got worse (A f2 1.1094 against 1.0841, A f3 1.0951 against 1.0692, B f3 1.1342 against 1.1218). In the folds where the base was anti-split, inner-CV improved strongly (A f0 1.1034 to 1.0774, A f1 1.0913 to 1.0415, B f0 1.1134 to 1.0758).

The train-only derive log shows the cause is the start point, not expressiveness:
- In s3 f2 and f3 the base already picks sk≥3 on 15/26 and 20/27 gated train shapes. The IQR-scaled bowl then pushes those shapes to sk 8, and train at w0 gets worse: f2 1.0822 to 1.1023, f3 1.0677 to 1.0806.
- The recipe's large-grid sign rule forces a nonzero sign even when train prefers (0,0), which puts CMA in a bad basin. At w0, train reaches 1.218 (A f0), 1.198 (A f2) and 1.133 (B f3). CMA with 2000 evaluations does not recover in A f2, A f3 and B f3.

**What this means for the cause.** C1 and C3 are real a6000 causes and not s0/s1 artefacts:
- On two more bases the same blocks, with train-formula constants, repair the large-grid class to about 1.05.
- They lift sk≥3 picks from 10 to 24 of 31 (s3) and from 5 to 21 of 31 (s2).
- They take about 0.045 off pooled holdout.
- The bowl stays a bowl in every fold.

H20A's larger inner-CV gain on s1 is partly base-specific. The failure mode is not the C5 hill of H20B. Instead, the recipe's w0 damages the start point in folds where the base already encodes split-K or where train says the large-grid block should start at zero, and the refit cannot climb back. This is consistent with H07b, which found that some evolved bases carry split-K elsewhere.
- **상수의 출처**: All constants were derived by /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/r3_H22/derive.py. It is a copy of r2_H20/derive.py with only the base rule changed; the diff is at the top of the file, and the log and values are in derive.log and derive.json in the same directory. It reads only _splits(t, fold=f, k=4, design='nkband').train.shapes and never touches holdout shapes or results.

**Gate P per fold.** Taken unchanged from an/logs/fold_gates.json, the H20A gates chosen on train:
- f0: log_flops >= 33.2131 and owps < 234057
- f1 and f3: log_flops >= 32.5 and owps < 234057
- f2: owps < 249661 and roofline_ratio >= 1.32805

**Centre c_f.** The median, over gated train shapes, of L at each shape's train-optimal config. It is 5.415 in every fold for both bases (L_opt depends only on shape and config), from 23 / 28 / 26 / 27 gated train shapes.

**Bowl strength b_f.** b_f = median over gated train shapes of rIQR(base score at base w0) divided by median rIQR((L - c_f)^2), where rIQR = IQR/1.349. The bowl's w0 = (-2·c_f·b_f, b_f).
- Run A (s3): rIQR 22.93 / 19.42 / 42.05 / 7.70 gives b_f 14.19 / 12.07 / 26.14 / 4.79.
- Run B (s2): rIQR 25190 / 352.9 / 18.40 / 71.36 gives b_f 15592 / 219.4 / 11.44 / 44.36.
- The quad spread is about 1.61 in every fold.

**Large-grid gate.** `p.roofline_ratio >= 1 and p.output_work_per_sm > 200000`, fixed from H20A/r09 and not tuned.

**Large-grid w0.** w0 = sign × robust scale:
- Robust scale = median rIQR(base + bowl score at w0) divided by median std of the term, over large-grid train shapes.
- Sign = the best all-nonzero combination on train at the std-ratio scale, with ties broken in itertools order, exactly as in H20.
- Run A w0: f0 [-95.5, -106.4], f1 [-81.8, -91.1], f2 [-168.9, -188.1], f3 [+31.1, -34.7].
- Run B w0: f0 [-10440, -11626], f1 [-1459, -1624], f2 [-75.6, -84.2], f3 [-282.8, -315.0].

**Base weights.** The existing w0 from rules/c2_evolved_s{3,2}.json, unchanged.

### 11. H23 — Margin-based bowl strength: works on the s1 base (Run B ties H20A), fails on the H18A/s0 base (Run A loses fold 0's split-K picks). Refuted by Run A.

- **시험한 원인**: C5 (weight scale) and how robust the current best rules are. Claim: the split-K bowl strength b needed is the score margin the bowl has to overturn at the base's own anti-split pick, not the base's overall spread. If so, one sweep-free train formula should reproduce H18A's swept b (s0 base) and H20A's IQR b (s1 base).
- **바꾼 것**: I kept the code, gates, extra features and every other starting weight of two existing rules (H18A and H20A). I replaced only the starting weights of the two split-K bowl terms. For each fold, on train shapes inside the split-K gate, I compared the base rule's own pick (bowl switched off) with the fastest split_k>=3 candidate. I then computed the smallest bowl strength that makes that candidate score better than the pick, and took the median over shapes as b. The bowl was set to b*(L-5.415)^2, which is the w0 pair (-2*5.415*b, +b). The official fit then refits everything from there.
- **돌린 것**: A_margin_H18A: inner 1.0830 · 홀드아웃 1.0936 / B_margin_H20A: inner 1.0775 · 홀드아웃 1.0697
- **판정**: refuted — The verdict is refuted: Run A triggers the refute clause, which applies to either run. Run B on its own meets all its confirm conditions. So the margin formula works on the s1 base but does not replace the train sweep on the s0-type H18A base.

**Before the official runs: formula b against the reference b (train only)**

Run A (H18A; reference is the swept b):
| Fold | Formula b | Swept b | Ratio | Qualifying shapes |
|---|---|---|---|---|
| f0 | 8.376 | 100 | 11.9x too small | 19/23 |
| f1 | 22.81 | 1 | 22.8x larger | 13/28 |
| f2 | 30328 | 3e5 | 9.9x too small | 24/26 |
| f3 | 0.659 | 1 | 0.66x | 12/27 |

Run B (H20A; reference is the IQR b):
| Fold | Formula b | IQR b | Ratio | Qualifying shapes |
|---|---|---|---|---|
| f0 | 2581 | 714.8 | 3.6x | 19/23 |
| f1 | 1.931 | 6.848 | 0.28x | 15/28 |
| f2 | 6.345 | 41.26 | 0.15x | 15/26 |
| f3 | 337.3 | 589.5 | 0.57x | 25/27 |

- No fold fell back to the IQR formula.
- The "within 10x of the swept b in f0 and f2" condition holds in f2 (9.9x) and fails in f0 (11.9x).

Frozen train check, train gm with the rest of w0 fixed. Columns: bowl off (b=0) / reference b / 0.5*b_f / b_f / 2*b_f.
| Run, fold | b=0 | Reference | 0.5*b_f | b_f | 2*b_f |
|---|---|---|---|---|---|
| A f0 | 1.1077 | 1.1171 | 1.1376 | 1.1495 | 1.1329 |
| A f1 | 1.0621 | 1.0634 | 1.0590 | 1.0590 | 1.0566 |
| A f2 | 1.0926 | 1.0477 | 1.0927 | 1.0919 | 1.0651 |
| A f3 | 1.0634 | 1.0651 | 1.0658 | 1.0651 | 1.0685 |
| B f0 | 1.1093 | 1.1040 | 1.0982 | 1.0860 | 1.0801 |
| B f1 | 1.0710 | 1.0759 | 1.0692 | 1.0707 | 1.0759 |
| B f2 | 1.0599 | 1.0585 | 1.0601 | 1.0568 | 1.0527 |
| B f3 | 1.1318 | 1.1188 | 1.1415 | 1.1274 | 1.0995 |

For Run A this already showed the formula too weak in f2 (6/26 gated shapes pick sk>=3, against 23/26 at the swept b) and harmful in f0.

**Run A (H18A base) against its criteria**
- Inner-CV: 1.0830 against the 1.0773 bar. Fails; it is 0.0107 worse than H18A's 1.0723.
- Holdout: 1.0936 against the 1.0853 bar. Fails; H18A has 1.0753 and the vendor 1.0815.
- sk>=3 on hard shapes: 19/31 against the 24 bar. Fails; H18A has 27/31.
- Squared weight positive: yes, in 4/4 folds (fitted 22.2 / 22.8 / 57652 / 1.82).
- Hungry class (m1 classes): 1.102, against 1.059 for H18A.
- Refute clause: inner is at least 0.01 worse and hungry is at least 1.09, so it is met. The inner margin (0.0107) is only just over the line. Hungry, sk>=3 and holdout all move clearly the wrong way, though.

Where Run A loses:
- **Fold 0.** Holdout is 1.1321 against 1.0893. 9 of 11 hard holdout shapes now pick sk=1; H18A had 1 of 11. For example, 384x4096x4096 goes from 1.074 to 1.507, 1000x4096x4096 from 1.000 to 1.225, and 768/1500/1536/2048x4096x4096 from about 1.00-1.07 to 1.09-1.27. CMA raised b only from 8.4 to 22.2, against H18A's fitted 220.7.
- **Fold 1.** CMA did not move from w0 at all (moved=False). Holdout is 1.1566 against 1.1399.
- **Folds 2 and 3.** f2 is fine even though its start b was 10x too small, because CMA refit it to 57652: 1.0462 against 1.0490. f3 is 1.0417 against 1.0257.

**Run B (s1 base) meets every confirm condition**
- Inner-CV: 1.07745 against the 1.0821 bar. H20A has 1.07711, so this is a tie.
- Holdout: 1.0697 against the 1.0810 bar. H20A has 1.0710, also a tie.
- Squared weight positive: yes, in 4/4 folds.
- It beats the vendor in 3 of 4 folds (f0, f2, f3; f1 still loses at 1.145) and picks sk>=3 on 28/31 hard shapes.
- By class against H20A: hungry 1.061 (1.064), large-grid 1.060 (1.087; f2 8192x11008x4096 goes from 1.328 to 1.030), mid-easy 1.154 (1.129; f1 128x4096x256 goes from 1.09 to 1.36).
- The per-fold changes offset each other. f0 inner improves from 1.0742 to 1.0477 with b 3.6x larger. f3 inner worsens from 1.0898 to 1.1227 with b 0.57x.
- Selected by inner-CV: Run B, 1.0775 against 1.0830.
- Pooled holdout 1.0697 is below the vendor's 1.0815 by 0.012, about the noise level. It is not a new best on inner-CV.

**What this says about C5**
The margin idea is partly right. Unlike the IQR formula in H20B, it never produced a start that the fit flipped into a hill (all 8 fitted squared weights are positive). But it does not remove the need for a sweep on the s0-type base, for two reasons:
1. **A median of single-pair margins underestimates the strength needed.** Once the base pick i loses to j, other non-split or wrong-tile candidates take over. A train-only diagnostic (diag.log, used for explanation only, not for selection) finds the frozen-train best b at 24.7x b_f in A f2, above every shape's margin. In B f0 and B f3 it sits above every margin (2.9x and 5.3x b_f).
2. **On H18A, the margin is measured at a w0 that is far from the fitted weights.** On f0, the frozen w0 base has train 1.108, while the fitted H18A reaches 1.049. No bowl strength improves the frozen f0 base: its best b is about 0 and b_f sits near the worst point. So the margin describes a base that CMA later changes a great deal.

On the s1 base, the w0 spread and the margin agree within 0.15-3.6x, so both formulas work there.
- **상수의 출처**: All constants come from each fold's TRAIN shapes only: _splits(t, fold=f, k=4, design='nkband').train.shapes. derive.py never reads holdout shapes or results.

- **Bowl centre c = 5.415037.** This is the median over gated train shapes of L at the train-optimal config. It was recomputed per fold and asserted equal to it/r2_H20/derive.json: 23/28/26/27 gated shapes.
- **Gates.** These are the existing fold gates already in the base code (an/logs/fold_gates.json). They are unchanged.
- **b_f.** On gated train shapes: i = top_k(s_base, 1), with s_base being the base code at its w0 with the two L weights set to 0. j = the fastest train candidate with split_k>=3. For shapes where i != j and (L_i-c)^2 > (L_j-c)^2: b_shape = (s_base[j]-s_base[i]) / ((L_i-c)^2 - (L_j-c)^2). Lower score is preferred, since top_k keeps the smallest. b_f is the median of b_shape. Additivity was asserted: s_base + gated L-terms reproduces the full code at the original w0.
- **Run A values.** Indices w[28,29] / w[34,35] / w[41,42] / w[31,32]. b_f = 8.3755 / 22.8106 / 30328.29 / 0.659027. They replace (-1100,100) / (-12,1) / (-3.3e6,3e5) / (-11,1).
- **Run B values.** Indices w[17,18] / w[29,30] / w[35,36] / w[40,41]. b_f = 2581.49 / 1.93133 / 6.34454 / 337.274. They replace the IQR b_f 714.83 / 6.848 / 41.263 / 589.48.
- **w0 pair.** w0 = (-2*c*b_f, b_f), rounded to 8 decimals.

Everything else is copied unchanged from H18A_1cta.json and H20_A_s1.json. That includes the other w0 values, which were derived on train in earlier rounds.

One deviation, as the hypothesis prescribed: c = 5.415 replaces H18A's implicit centres 5.5 (f0, f2, f3) and 6.0 (f1).

### 12. H24 — Control for C3: split-only placebo axes in H18A's large-grid branch still rescue the big grids (refuted as written), but the placebo moves no picks; the rescue uses 1-CTA tiles picked through H18A's other residency/register terms, and the same placebo on H01b loses the whole gain

- **시험한 원인**: CONTROL for C3's mechanism. Rules cannot see swizzle, so the large-grid gain should come from a tile-choice route: 1-CTA 128x256 or 256x128 tiles whose swizzle-1 variant is near optimal. If that is right, a branch with the same gate, weight slots and start scaling, but with axes that carry no tile-size or residency information, should lose the whole large-grid gain. If the gain survives, it came from branch capacity or refit luck.
- **바꾼 것**: In H18A's large-grid branch (gate unchanged: roofline_ratio >= 1 and owps > 2e5), I swapped the two axes that can tell tiles apart (resident_block_scarcity, register_file_pressure) for two split-K-only axes (serial_reduction_dependency, parallel_reduction_traffic_fraction). The weight slots stay the same: (30,31), (36,37), (43,44), (33,34). Their start weights come from H01b's robust-spread recipe on train shapes only. Everything else in H18A is untouched. As a second run, I made the identical swap on H01b, which is H18A without its band-P single_resident_cta term. This checks whether that term gives the fit a second way to reach 1-CTA tiles.
- **돌린 것**: H24_placebo (faithful: H18A base, placebo branch): inner 1.0821 · 홀드아웃 1.0796 / H24B_placebo_H01b (same placebo on H01b = H18A without the band-P 1-CTA indicator): inner 1.0888 · 홀드아웃 1.0930
- **판정**: refuted — Inner-CV selects H24_placebo (1.0821, against 1.0888 for H24B). H24_placebo is also the faithful run. Holdout is 1.0796 (vendor 1.0815, H18A 1.0753), 2 of 4 folds beat the vendor, and sk>=3 is picked on 17 of 31 hard shapes.

**Pre-registered criteria on the selected run: refute fires, confirm fails.**
- The big class (owps > 2e5, n=9) is 1.054. H18A has 1.060 and L1 1.142. That is <= 1.075, so refute is met.
- The three 8192-row trap shapes are all rescued, on 1-CTA swizzle-1 picks. All are <= 1.10, so refute is met again:
  - 8192x11008x4096: 1.030 (256x128)
  - 8192x12288x4096: 1.019 (128x256)
  - 8192^3: 1.033 (128x256)
- The confirm conditions all miss:
  - Big class >= 1.11: no.
  - Trap shapes >= 1.25: no.
  - Pooled >= 1.080: 1.0796, just under.
  - 2048-row shapes back to about 1.013: no, they are 1.090 (f2 2048x11008x4096) and 1.087 (f3 2048x12288x4096).

So the claim as written fails. On H18A, a branch whose axes cannot tell tiles apart does not lose the large-grid gain.

**What actually carries the gain. These are post-hoc diagnostics, not used for selection.**

1. **Branch capacity is ruled out.** The fitted placebo branch moves no pick. Zeroing its two fitted weights changes 0 picks on the gated train shapes in all 4 folds, and 0 on the holdout shapes. At the start weights the placebo was already inert: +k gives the same train regret as 0 in every fold, and -k is worse.
   - The placebo axes carry no tile information. Within-shape r on gated train shapes, the same in all folds:
     - serial_reduction_dependency: +0.029 with single_resident_cta and -0.019 with log2(tile area); max per-shape |r| is 0.032.
     - parallel_reduction_traffic_fraction: 0.000 with both.
     - No fold is above 0.2.

2. **The rescue is a refit of H18A's own residency and register weights. It is still the tile route, just not gated.**
   - **f2:** the compute-bound register_file_pressure weight w[9] went from +200.7 to -8.87. Every other refit of this base raised it instead: s0 702.9, H01b 419.5, H18A 530.1, L1 unchanged. Moving only this weight from start to fitted turns 6 of 6 gated train shapes into 1-CTA picks. Reverting only it sends them back to the trap (gated train gm 1.236).
   - **f3:** the ungated resident_block_scarcity weight w[3] went from 11.06 to 5.56. The other refits gave 27.2, 17.1, 12.6 and 21.2. Moving only this weight gives 5 of 5 1-CTA picks.

3. **The fit can relax those global penalties because H18A's band-P single_resident_cta term holds the hungry shapes at 2-CTA.** With H24's fitted weights, zeroing that term makes the P-gated train shapes pick 1-CTA in:
   - f2: 26 of 26
   - f3: 25 of 27
   - f0: 17 of 23
   - f1: 8 of 28

   So "prefer 1-CTA everywhere, penalise it inside P" gives large grids the same tiles the branch was giving them.

4. **The same placebo on H01b, which lacks that term, loses the whole gain. It meets every confirm criterion.**
   - Big class 1.140 (L1 1.142, s0 1.141).
   - The trap shapes are 1.330, 1.334 and 1.436, all on 2-CTA 128x128x32 st3 swizzle-1.
   - Both 2048-row shapes are back at 1.013.
   - Pooled 1.0930, with 1 of 4 folds beating the vendor.
   - Consistent with this, zeroing the real branch in H01b's and H18A's fitted weights sends the f2/f3 large grids back to the trap. For example, H01b f3 8192^3 goes from 1.037 to 1.436, and H18A f2 8192x11008x4096 from 1.033 to 1.330.

5. **Side effects are refit noise, not the placebo.**
   - H24's f0 fit ended far worse than H18A's from a near-identical start (train 1.094 against 1.049). It picks sk>=3 on 0 of 11 hungry holdout shapes (f0 hungry gm 1.131), which is why hungry is 1.079 against H18A's 1.059. Mid-easy is 1.163.
   - In H24B's f2 fit, CMA never left its start point (0 of 45 weights changed).

**What this means for C3.** The tile-choice mechanism holds up. Every large-grid rescue in all five rules compared is a 1-CTA swizzle-1 pick chosen through residency or register axes, and a base with no other route to those tiles loses the gain. What needs revising is the claim that a gated large-grid branch is needed. On H18A the same route can come from global pro-1-CTA weights plus the band-P 1-CTA penalty. Whether a refit finds that route depends on the fit path: the H24 refit did, while the s0, L1, H01b and H18A refits pushed those weights the other way.

Process note: run 1 was started under a 25-minute shell timeout. I removed the wrapper mid-run so the same process could finish; nothing was re-run.
- **상수의 출처**: - **Gate:** roofline_ratio >= 1 and owps > 2e5, copied unchanged from H18A / r09.
- **Weight slots:** unchanged from H18A.
- **Placebo start weights, per fold, from TRAIN shapes only** (prep.py, prep.log, prep.json). k = median over gated train shapes of the within-shape IQR/1.349 of H18A's score (at its w0, with the branch weights set to 0), divided by the median within-shape std of the axis.
  - Median axis std, the same in every fold: 0.04927 for serial_reduction_dependency, 0.4714 for parallel_reduction_traffic_fraction.
  - Median base spread: f0 0.5969, f1 20.27, f2 5493, f3 4.897.
  - Resulting w0: f0 12.114 / 1.266; f1 411.47 / 43.01; f2 111493.5 / 11654.0; f3 99.39 / 10.39.
- **Sign:** +1 for both axes in all folds. +k gave the same frozen train regret as 0 and -k was worse, so +k was the lower of the two.
- **Gated train shapes per fold:** 8, 8, 6 and 5.
- **All other constants** are H18A's w0 as published, which are themselves train-derived.
- **Variant B:** the same recipe on H01b (prep_b.py, prep_b.log) gave the identical k and signs, because the band-P term does not touch the large-grid gated shapes.
- **Correlation diagnostic:** within-shape r of each placebo axis with single_resident_cta and with log2(tile_m*tile_n) on gated train shapes only.
- **Post-hoc attribution** (diag_train.py, diag_attr.py, diag_src.py, post.py) uses fitted weights. The train parts read train shapes only. The holdout breakdowns are reporting only and were not used for any selection.

### 13. H25 — Control: renaming the weights (identical objective) moves inner-CV by about 0.01 and swaps the order of the two current bests; H18A's fold-0 fit reproduces, so it was not a lucky draw

- **시험한 원인**: C7 (new): the fitted result depends on the CMA/polish path by more than the ~0.01 noise assumed so far. Sub-claim: H18A's fold-0 fit (train 1.049) was a lucky draw, and the H21A/H24 fold-0 collapse (train 1.094) is path noise, not the code change.
- **바꾼 것**: The rules themselves were not changed. In each fold I renamed the weights with a fixed random shuffle (w[i] becomes w[pi[i]], with the start values moved to match). The fitter therefore sees exactly the same problem from the same start, but its random draws and its coordinate-polish order land on different weights, so it takes a different path. Checks: the scores are bit-identical to the originals on every train shape, about 2.93M candidates per base, both at w0 and at a random w. A train-only refit of the unrenamed rules reproduces the original fitted weights bit for bit in 8 of 8 base-folds. So every difference below comes only from the fitting path.
- **돌린 것**: H25A_perm_H18A (H18A_1cta, weight labels permuted, rng 1000+f): inner 1.0821 · 홀드아웃 1.0693 / H25B_perm_H23B (H23B_margin_H20A, weight labels permuted, rng 1000+f): inner 1.0691 · 홀드아웃 1.0602
- **판정**: confirmed — **Verdict: confirmed.** Both bases meet the confirm test, and both break the refute test. The H18A-specific sub-claim is not supported.

- **Confirm test:** |Δ inner| is at least 0.008 in both bases (A +0.0098, B −0.0084). B also has a fold whose holdout moves by at least 0.03 (f1, −0.042).
- **Refute test:** it needed Δ inner ≤ 0.004 and Δ holdout ≤ 0.006. Both bases fail it.
- **Not supported:** the fold-0 collapse does not reproduce when only the labels change.

**Selection.** Inner-CV picks B (1.0691 vs 1.0821). This is **not a new rule**: it is H23B with its weights renamed. Its lower inner-CV (1.0691) and holdout (1.0602, both the lowest recorded) are one path draw of a rule already on record. Averaged over the two draws, H23B scores inner 1.0733 / holdout 1.0650.

**Run A, H18A: original → permuted (Δ)**

| Fold | train | inner | holdout |
|---|---|---|---|
| f0 | 1.0493 → 1.0485 (−0.0009) | 1.0690 → 1.0729 (+0.0039) | 1.0893 → 1.0638 (−0.0255) |
| f1 | 1.0599 → 1.0621 (+0.0023) | 1.0783 → 1.1133 (**+0.0350**) | 1.1399 → 1.1394 (−0.0004) |
| f2 | 1.0440 → 1.0423 (−0.0017) | 1.0528 → 1.0487 (−0.0041) | 1.0490 → 1.0494 (+0.0004) |
| f3 | 1.0567 → 1.0575 (+0.0008) | 1.0895 → 1.0945 (+0.0050) | 1.0257 → 1.0284 (+0.0027) |
| Pooled | – | 1.0723 → 1.0821 (+0.0098) | 1.0753 → 1.0693 (−0.0060) |

- sk≥3 on hard shapes: 27/31 → 29/31. Folds beating the vendor: 3 → 3.
- Fold-0 hungry holdout shapes picking sk≥3: 10/11 → 10/11.

**Run B, H23B: original → permuted (Δ)**

| Fold | train | inner | holdout |
|---|---|---|---|
| f0 | 1.0426 → 1.0403 (−0.0022) | 1.0477 → 1.0444 (−0.0033) | 1.0586 → 1.0588 (+0.0002) |
| f1 | 1.0516 → 1.0586 (+0.0070) | 1.0687 → 1.0782 (+0.0096) | 1.1449 → 1.1033 (**−0.0416**) |
| f2 | 1.0443 → 1.0471 (+0.0028) | 1.0715 → 1.0812 (+0.0097) | 1.0369 → 1.0350 (−0.0019) |
| f3 | 1.0403 → 1.0417 (+0.0014) | 1.1227 → 1.0723 (**−0.0504**) | 1.0427 → 1.0450 (+0.0023) |
| Pooled | – | 1.0775 → 1.0691 (−0.0084) | 1.0697 → 1.0602 (−0.0096) |

- sk≥3 on hard shapes: 28/31 → 26/31.
- Fold-0 hungry holdout shapes picking sk≥3: 10/11 → 10/11.

**What this means for C7**
1. **The fitting path alone moves inner-CV by about 0.01.** Same objective, start and budget, only a different CMA/polish path: pooled inner moves 0.008–0.010 and pooled holdout 0.006–0.010. Treating those two draws per rule as a rough estimate, one run has sd ≈ 0.006–0.007, so the difference between two single runs has sd ≈ 0.009. Pooled inner-CV gaps below about 0.02 are therefore ties.
2. **Per-fold numbers are much noisier.** Per-fold inner moves up to 0.035 (A f1) and 0.050 (B f3). Per-fold holdout moves up to 0.026 (A f0) and 0.042 (B f1).
3. **Train gm barely moves (at most 0.007 in any fold), but the fitted weights do.** The median relative difference between the two fitted weight vectors is 0.52–0.87. The fit finds equally good train optima at very different weights, and those generalise differently.
4. **The ranking of the two current bests flips.** Originally H18A was ahead (1.0723 vs 1.0775). After renaming, H23B is ahead (1.0691 vs 1.0821). Averaged over the two draws, H23B is at 1.0733 and H18A at 1.0772, a gap of 0.004. That is a tie, and it bears directly on H28, which picks bases this way.

**H18A fold 0 was not a lucky draw.** The renamed fit reaches train 1.0485 against 1.0493, with 10/11 hungry holdout shapes still at sk≥3. The fitted bowl stays a bowl; its vertex (the log2 iterations-per-split where the bowl is lowest) moves only 5.06 → 4.95. The H21A/H24 collapse (train 1.094, 0/11) does not reproduce from a relabelling, so it most likely comes from those rules' code changes rather than a general path lottery. Two draws cannot rule out that a minority of paths collapse.

**Several earlier per-shape blames were path noise.** These shapes move by large amounts with no code change:
- H18's 128x4096x4096 at 1.672 comes back to 1.080 (f0 holdout −0.026).
- H20A/H23B's 512^3 at 1.656 comes back to 1.091.
- H23B f1 4096x4096x11008 goes from 1.394 to 1.082.
- H23's f3 inner 1.1227, which was blamed on b being 0.57x, comes out at 1.0723.

The mid-easy class carries most of this noise: A 1.192 → 1.143, B 1.154 → 1.107. The hungry class moves only 1.059 → 1.061 and 1.061 → 1.058. The big class moves 1.060 → 1.060 and 1.060 → 1.065. So class-level claims about hungry and big shapes survive, while claims resting on single mid-easy outliers or on per-fold inner gaps do not.

Picks that changed (tile, split_k or regret): 16/65 shapes in A, 26/65 in B.

**Fitted weights mapped back to the original labels (original → permuted)**

H18A split-K bowl, (linear, squared) weight and vertex:

| Fold | Original | Permuted | Start w0 |
|---|---|---|---|
| f0 | (−2234, 220.7), vertex 5.06 | (−1041, 105.2), vertex 4.95 | (−1100, 100) |
| f1 | (−11.83, 2.117), vertex 2.80 | (−14.74, 2.361), vertex 3.12 | – |
| f2 | (−9.09e6, 8.60e5), vertex 5.28 | (−1.32e7, 1.23e6), vertex 5.38 | – |
| f3 | (−43.3, 4.36), vertex 4.97 | (−12.0, 1.33), vertex 4.52 | – |

H18A 1-CTA term (single_resident_cta):

| Fold | w0 | Original fit | Permuted fit |
|---|---|---|---|
| f0 | 351 | 245 | 1173 |
| f1 | 47.9 | 142 | 73.6 |
| f2 | 1.02e6 | 5.48e6 | 2.23e5 |
| f3 | 11.8 | 7.71 | 3.36 |

The 1-CTA term stays positive in every fold but its size changes up to 25x, so the magnitudes are poorly identified.

H23B split-K bowl (this rule has no 1-CTA term):

| Fold | Original | Permuted |
|---|---|---|
| f0 | (−35604, 3727), vertex 4.78 | (−112894, 11438), vertex 4.94 |
| f1 | (−20.99, 2.079), vertex 5.05 | (−25.13, 2.011), vertex 6.25 |
| f2 | (−257.8, 25.38), vertex 5.08 | (−116.9, 12.57), vertex 4.65 |
| f3 | (−11689, 1286), vertex 4.54 | (−11600, 1262), vertex 4.60 |

All 16 fitted bowls stay bowls (positive squared weight).
- **상수의 출처**: No new constants were added. Code, gates (log_flops ≥ 33.2131 or 32.5; output_work_per_sm < 234057 or > 200000; roofline_ratio), the bowl form, w0 values and extra_features are copied verbatim from H18A_1cta.json (s0 base + H01b terms + H18's train-scaled single_resident_cta) and from H23B_margin_H20A.json (s1 base + H20A blocks + H23's train margin b). Their provenance is as documented in H18, H20 and H23. The only new element is the relabelling: pi = numpy.random.default_rng(1000+f).permutation(len(w0_f)), as the hypothesis prescribes, stored in perms.json. new_w0[pi[i]] = w0[i], and the code has w[i] replaced by w[pi[i]]. Exact equality was verified on all train shapes: max |diff| is 0 at w0 and at a random w, in every fold of both bases. A train-only refit of both unpermuted rules reproduces the original RESULT's fitted w bit for bit in 8 of 8 base-folds, so the pipeline is deterministic and unchanged. Holdout was not used for anything except the reported official numbers.

### 14. H26 — A global split-K traffic penalty cuts over-splitting but moves the error to under-splitting and loses the hungry class on both bases (refuted)

- **시험한 원인**: C1 refined. The split-K bowl has the same centre for every shape, so nothing makes a split cost more on a large output. The hypothesis: one positive weight on a shape-aware traffic ratio (split-K reduction bytes over compulsory bytes) stops one-step over-splits of large outputs, and it does not pull thin-output hungry shapes back to split_k 1.
- **바꾼 것**: I wrote one new axis, split_reduction_traffic_ratio. It is the extra DRAM bytes a split-K reduction moves, divided by the bytes the GEMM must move anyway. Serial mode counts 2*(sk-1)*M*N*4 bytes and parallel mode counts 2*sk*M*N*4 bytes; the compulsory bytes are (MK+KN+MN)*2. The axis is 0 at split_k=1. I added it as one global, ungated term `s = s + f.split_reduction_traffic_ratio * w[n]` just before `return s` in every fold of two bases: H23B (Run A) and H18A (Run B). The start weight was set per fold from train shapes only, using the robust-spread ratio followed by the halving guard. CMA then refit all weights as usual.
- **돌린 것**: A_srt_H23B (H23B + global split_reduction_traffic_ratio): inner 1.0844 · 홀드아웃 1.1068 / B_srt_H18A (H18A + global split_reduction_traffic_ratio): inner 1.0854 · 홀드아웃 1.1066
- **판정**: refuted — **Selection.** Inner-CV picks Run A: 1.0844 against 1.0854 for B, which is a tie. Both runs are worse than their bases:
- A: inner 1.0844 against H23B's 1.0775 (+0.0070); holdout 1.1068 against 1.0697.
- B: inner 1.0854 against H18A's 1.0723 (+0.0131); holdout 1.1066 against 1.0753.

Neither beats the vendor's pooled 1.0815. A beats the vendor in 2 of 4 folds and B in 1 of 4.

**Refute check: fires in both runs.** The clause that fires is "under-split gap rises by >= 0.008". I measured the gaps with the base-number definition: sum of log regret over holdout picks with regret > 1.03, split by sk_pick against sk_best, divided by 65. This reproduces the bases exactly (H23B 0.0299, H18A 0.0311).

| Run | Over-split gap (base → run) | Change | Under-split gap (base → run) | Change |
|---|---|---|---|---|
| A | 0.0299 → 0.0197 | -0.0102 | 0.0115 → 0.0608 | +0.0493 |
| B | 0.0311 → 0.0184 | -0.0127 | 0.0126 → 0.0559 | +0.0433 |

The other three refute clauses do not fire:
- The fitted w[n] is positive in 3/4 folds in A and 4/4 in B, so the "w[n] <= 0 in >= 3 folds" clause fails.
- The over-split gap falls by more than 0.004 in both runs.
- A's inner-CV is worse than H23B's by 0.0070, which is under the 0.01 line.

**Confirm check, selected run A.**
- Over-split gap falls by >= 0.008: met (-0.0102).
- Fitted w[n] > 0 in >= 3 folds: met. Values are f0 -0.045, f1 +3.43, f2 +7.43, f3 +5.79.
- 512^3 <= 1.10: met (1.656 → 1.091).
- 4096^3 <= 1.10: missed. It stays at 1.344 (128x128x32 sk4), because the f0 weight fell to about 0.
- Hungry class <= 1.065: missed (1.152).
- Headline (inner < 1.0723 and holdout < 1.0697): missed.

**Named shapes.** Each cell is regret [pick]; the bases are H23B for A and H18A for B.

| Shape | Base A | Run A | Base B | Run B |
|---|---|---|---|---|
| 4096^3 | 1.344 [sk4] | 1.344 [sk4] | 1.344 [sk4] | 1.123 [128x256 sk1] |
| 3000x4096x4096 | 1.219 | 1.219 | 1.219 | 1.064 [128x256 sk1] |
| 4096x4096x11008 | 1.394 [sk6] | 1.055 [sk2] | 1.134 [sk8] | 1.109 [sk6] |
| 512^3 | 1.656 [sk8] | 1.091 [sk1] | 1.818 [sk8] | 1.818 [sk8] |
| 256x4096x4096 | 1.071 [sk1] | 1.071 [sk1] | 1.091 [sk1] | 1.091 [sk1] |

**Classes** (m1_classgap definitions; the number in brackets is the pooled log gap against the vendor, per 65 shapes):

| Class | Base A | Run A | Base B | Run B |
|---|---|---|---|---|
| Hungry (n=31, vendor 1.069) | 1.061 | 1.152 (+0.0356) | 1.059 | 1.125 (+0.0241) |
| Mid-easy (n=13) | 1.154 | 1.122 (+0.0035) | 1.192 | 1.192 |
| Big | 1.060 | 1.056 | 1.060 | 1.060 |
| M<=32 | 1.014 | 1.018 | 1.012 | 1.011 |

**What happened.** The term does what it was built for on large outputs:
- f1 4096x4096x11008 and 512^3 are fixed, as shown above.
- f1 128x4096x256 goes from 1.364 to 1.091.
- f3 over-splits at sk12/16/6 on 1024x4096x8192, 1024x4096x16384 and 1024x12288x4096 go from about 1.01-1.03 to 1.002.

But within any one shape it is still a monotone "split less" axis, and one weight cannot separate "large output, too many splits" from "large output, needs 3-4 splits". The ratios per extra split are close: 4096^3 is 1.33 and wants sk3; 2048x4096x4096 is 1.0 and wants sk4. So the fit trades one error for the other:
- A f1: 1024x4096x11008 goes from 1.005 to 1.074, and 1024x4098x4096 from 1.009 to 1.086.
- A f3: 1024x4096x2048 goes to sk1 at 1.095.
- B f0: 9 hungry holdout shapes collapse to sk1, including 384x4096x4096 at 1.507 and 2048x4096x4096 at 1.244.

The train pre-check already showed this trade before any official run. The term was scored with the base frozen at its w0, so this only describes the start point. At the chosen w0[n], train over-split counts fell by 0-4 per fold while under-split counts rose by 0-8. For example, A f1 went 8/18 → 8/21 (over/under), B f2 3/14 → 1/22, and B f3 17/16 → 13/24.

**Refit side effects add to the damage.**
1. **A f3: one catastrophic pick from refit drift.** 2048^3 went from 1.101 to 16.545 on a register-spilling 256x128x32 st2 config (3966 spill bytes, sk1). The new term is 0 on both configs. The pick changed because the refit shrank the has_spill weight 3.1x relative to H23B's fit (745390 → 237349), while wave_count_deficit shrank only 1.66x. That one shape adds 0.043 to A's pooled log gap. As a post-hoc illustration only: with 2048^3 excluded, A's under-split rise is +0.0076 instead of +0.0493. So the move from over- to under-splitting is still there, just smaller.
2. **B f2: the start guard hit its cap.** It stopped at 8 halvings with the start still damaging train: frozen train gm 1.0963 against 1.0477.
3. **B f0: the known fold-0 fragility.** The fold-0 fit ended at train 1.1047 against H18A's 1.0493, the same collapse seen in H21, H23A and H24.

**What this means for C1.** Over-splitting large outputs is real, and making split cost grow with output size does fix it: the over-split gap fell by about 0.010-0.013 in both runs. But a penalty that grows monotonically with split_k is the wrong form. Refit on the whole train set, it trades over-splits for under-splits and gives up the hungry-class gain that was the whole point of the bowl. What C1 needs is a per-shape optimum location, not an extra one-sided push. That is consistent with the H06 control and with H02's finding that one fixed centre is wrong.
- **상수의 출처**: **The axis.** Pure physics from shape and config fields: p.M, p.N, p.K, p.acc_bytes_per_element, p.bytes_per_element, cfg.split_k and cfg.split_k_mode. It reads no hw fields and no time or answer columns. The formula is exactly the one the hypothesis gives. Every train shape in every fold varies in the term, with IQR > 0 in all 48-49. The maximum value is 34.9, inside the file's expected_range of [0, 64].

**w0[n].** Derived per fold from each fold's train shapes only (_splits(t, fold=f, k=4, design='nkband').train.shapes), with the base frozen at its own w0. The start value is the median over train shapes of the within-shape IQR/1.349 of the base score, divided by the median over train shapes of the within-shape IQR/1.349 of the term. The sign is fixed to +. The guard halves w0[n] while frozen-base train gm exceeds the term-off train gm by more than 0.002, with at most 8 halvings.

Each row reads: start value → halvings → w0[n], then train gm with the term at 0 → at w0[n], then the counts of train shapes with sk_pick > sk_best and sk_pick < sk_best at term 0 → at w0[n].

Run A (H23B base):
- f0: 2331.9 → 8 (cap) → 9.109; 1.0860 → 1.0900; over 4→3, under 26→26.
- f1: 7.188 → 3 → 0.8985; 1.0707 → 1.0712; over 8→8, under 18→21.
- f2: 35.27 → 3 → 4.409; 1.0578 → 1.0595; over 9→7, under 22→25.
- f3: 674.6 → 5 → 21.08; 1.1274 → 1.1276; over 3→2, under 26→28.

Run B (H18A base):
- f0: 26.85 → 1 → 13.43; 1.1171 → 1.1178; over 6→6, under 21→21.
- f1: 13.51 → 4 → 0.8442; 1.0634 → 1.0621; over 8→8, under 24→24.
- f2: 2.486e5 → 8 (cap) → 971.1; 1.0477 → 1.0963; over 3→1, under 14→22.
- f3: 3.439 → 1 → 1.719; 1.0651 → 1.0655; over 17→13, under 16→24.

**Fitted w[n]** (read from the RESULT.json files):
- A: f0 -0.0452, f1 +3.434, f2 +7.431, f3 +5.793.
- B: f0 +36.01, f1 +1.232, f2 +1091.6, f3 +2.126.

**Everything else.** All other constants (gates, bowl centres, base weights) are inherited unchanged from the H23B and H18A rule files. The 1.03 threshold in the over/under-split gap is copied from the definition behind the base numbers; it reproduces H23B 0.0299 and H18A 0.0311 exactly. No holdout shape or holdout result was used to set any constant or to choose between runs. The 2048^3 diagnostic and the "excluding 2048^3" numbers are post-hoc explanations only.

### 15. H27 — The start-point guard removes H22's inner-CV loss on s3 folds 2-3, but it does so by switching the blocks off, and holdout gets worse in both runs (inconclusive)

- **시험한 원인**: C5, stated for the start point. The claim: H22's inner-CV loss in the folds where the base already splits (s3 f2/f3, s2 f3) came from a bad w0. Two things made it bad: the IQR-scaled bowl was too strong for a base that already splits, and the large-grid sign search could not return (0,0). A train-only guard on w0, with the same code, should remove that loss.
- **바꾼 것**: I kept H22's rule code, gates, bowl centre (5.415) and extra features exactly as they were. Only the starting weights of the four added terms changed: the two bowl weights and the two large-grid weights. For each fold, using training shapes only, I tried all 9 sign choices for the large-grid weights, including zero, and kept the one with the lowest training regret (on a tie, the one with more zeros). Then, while the rule at its starting weights was still more than 0.002 worse on training shapes than the bare base rule, I halved all four added weights together, at most 8 times.
- **돌린 것**: H27A_s3_guard (H22_A_s3 + sign search incl. zero + halving guard): inner 1.0673 · 홀드아웃 1.0993 / H27B_s2_guard (H22_B_s2 + sign search incl. zero + halving guard): inner 1.0769 · 홀드아웃 1.1053
- **판정**: inconclusive — **Selection.** Inner-CV selects Run A: 1.0673, against 1.0769 for B. That is the lowest inner-CV recorded so far (H18A 1.0723). But A's pooled holdout is 1.0993, worse than the vendor's 1.0815 and worse than H22A's 1.0826. It beats the vendor in 0 of 4 folds. B's holdout is 1.1053, beating the vendor in 1 of 4 folds (H22B 1.0986).

**What the guard did (train only, derive.log).** Every fold's w0 changed, so no fold is an unchanged control.
- Signs only, no halving (4 folds):
  - A f0: (0,0). It tied at train 1.0821 with H22's (-1,-1).
  - A f1: (-1,0). It tied at 1.0404 with H22's signs.
  - B f0: (0,0), 1.0970, better than H22's 1.1030.
  - B f2: (0,-1). It tied at 1.0569 with H22's signs.
- The guard fired (4 folds):
  - A f2: 8 halvings. Train at w0 went 1.1982 (H22) → 1.1023 (sign (0,0)) → 1.0819. G_base is 1.0822. The bowl went from 26.1 to 0.10.
  - A f3: 8 halvings, which is the cap. Train went 1.0828 → 1.0806 → 1.0707, still above G_base+0.002 = 1.0697. The bowl went from 4.79 to 0.019.
  - B f1: 4 halvings. Train went 1.1062 → 1.0866 → 1.0791 (G_base 1.0800). The bowl went from 219 to 13.7.
  - B f3: 7 halvings. Train went 1.1328 → 1.0992 → 1.0789 (G_base 1.0777). The bowl went from 44.4 to 0.35.
- A correction to the premise. H22's forced nonzero large-grid sign hurt the actual w0 only in A f2 (1.198) and B f1/B f3 (1.106/1.133). The 1.218 quoted for A f0 was measured at H22's std-ratio scale, not at the robust scale its w0 actually used. At its real w0, A f0 was 1.0821, the same as with sign (0,0).

**Confirm criteria.**
- Mechanism: met. Against the paired H22 run, mean inner-CV improvement over the changed folds is 0.0135 in A (all 4 folds) and 0.0160 in B. Over the guard-fired folds alone it is 0.0255 in A and 0.0224 in B.
- "Unchanged folds move by < 0.005" is vacuous, because no fold stayed unchanged. The sign-only folds moved inner by -0.0076, +0.0048, -0.0186 and -0.0006.
- Run A clause: met.
  - s3 f2: 1.1094 → 1.0838. The bar was ≤ 1.095.
  - s3 f3: 1.0951 → 1.0696. The bar was ≤ 1.083.
  - Both improved by 0.026, above the required 0.012.
  - f0 improved by 0.0076, and f1 got worse by only 0.0048.
  - Pooled inner 1.0673, against the ≤ 1.0746 bar.
- Headline: inner < 1.0723 is met (1.0673), but holdout < 1.0697 fails badly (1.0993).

**Refute criteria.** Neither fires. The guard-changed folds improved by far more than 0.005, and s3 f2/f3 did change.

**Why inconclusive rather than confirmed.** The inner-CV gain comes from turning the new blocks off, not from giving them a better start.
- In A f2, CMA never left the guarded start: 0 of 45 weights moved. In A f3, only 1 of 40 moved.
- The reason is the fitter's step size, which is 0.6 × max(|w|, 1). A bowl halved to 0.1 or 0.02, or a large-grid weight set to 0, cannot grow back to the scale it needs (tens to hundreds). The zero-start large-grid weights ended at |w| ≤ 10 in every fold. H22's fits ended at 50-500 on s3 and up to 3e4 on s2.
- So A f2/f3 simply reproduce the bare s3 base: inner 1.0838/1.0696 against the base's 1.0841/1.0692, and train 1.0819/1.0681 against 1.0817/1.0677.

The holdout goes the other way.
- A per fold, against H22A:
  - f0: +0.023. This fold only had its large-grid sign zeroed, which was inert on train. After the refit, hungry shapes landed on 1-CTA 256x128 tiles: 512x4096x4096 went from 1.000 to 1.133, and 2048x4096x4096 from 1.002 to 1.133.
  - f1: +0.013.
  - f2: +0.026. For example, 128x4096x1024 went from 1.043 to 1.304.
  - f3: +0.005. The swizzle-trap rescue was lost: 8192^3 went from 1.033 to 1.436, and 8192x12288x4096 from 1.018 to 1.334.
- B per fold: f3 worsened by +0.027 (8192^3 from 1.033 to 1.413), and only f2 improved (-0.015).
- In 6 of 8 folds, inner-CV and holdout moved in opposite directions.

**Classes (m1 classes), H22 → H27.**

| Class | A: H22 → H27 | B: H22 → H27 |
|---|---|---|
| Large-grid | 1.050 → 1.112 | 1.052 → 1.092 |
| Hungry | 1.100 → 1.099 | 1.097 → 1.085 |
| Mid-easy | 1.126 → 1.158 | 1.220 → 1.247 |
| M≤32 | 1.019 → 1.031 | 1.018 → 1.028 |

The large-grid loss comes entirely from f3. On hard shapes, sk≥3 is picked on 26/31 in A (H22A 24) and 23/31 in B (H22B 21).

**Against fit-path noise.** H25 relabelled weight indices, which leaves the objective unchanged. That alone moved per-fold inner by a mean |d| of 0.012 and 0.018, with a maximum of 0.050. The sign-only folds here moved by a mean of 0.008. The four guard folds all moved in the same direction by 0.010-0.035. That is consistent with a real start-point effect, but it is only about 1.5-2 times the noise.

**What this means for C5.** The start point was indeed why H22's s3 f2/f3 inner-CV fell below the base. But the only train-only fix found here is to shrink the blocks until they are off. Inner-CV then rewards the bare base in exactly the folds where H22's blocks earned their holdout gains. A shrink-by-halving guard cannot tell "bad start" from "block not needed on train". Because the fitter's step size is set relative to |w|, a small start also freezes the block.
- **상수의 출처**: - **Unchanged from H22** (it/r3_H22/derive.py and derive.json, all train-only): the code, the gates, the bowl centre c_f = 5.415, the bowl strength b_f, and the large-grid magnitudes |w0|.
  - c_f is the median of L at the train-optimal config over the gated train shapes.
  - b_f is the median within-shape IQR/1.349 of the base score divided by that of (L - c_f)^2.
  - The large-grid magnitudes are the robust-std ratio.
- **New in H27** (it/r4_H27/derive.py). It uses only _splits(t, fold=f, k=4, design='nkband').train.shapes of each fold, and train gm is computed exactly as the probe computes it (evaluate_scores top-1, same union feature registry). Holdout shapes and results were never read.
  - (1) Signs: argmin of train gm at w0 over the 9 (sa, sb) pairs. Ties (|diff| < 1e-12) go to fewer nonzero entries, then to the pair closest to H22's signs, then to itertools order.
  - (2) G_base: train gm of rules/c2_evolved_s3.json or c2_evolved_s2.json at its own w0, same fold. Values: A 1.0891 / 1.0860 / 1.0822 / 1.0677, B 1.1019 / 1.0800 / 1.0622 / 1.0777.
  - (3) Guard: halve all four new-block entries while train gm > G_base + 0.002, at most 8 times. The tolerance and the cap are the fixed values from the hypothesis.
- **Result per fold (signs, halvings):** A (0,0) 0 / (-1,0) 0 / (0,0) 8 / (0,0) 8 (cap reached); B (0,0) 0 / (0,0) 4 / (0,-1) 0 / (0,0) 7. The vertex stays at 5.415 in every fold.

### 16. H28 — Per-fold inner-CV choice of base: both composites reproduce exactly and meet every number target (A 1.0563, 3/4 folds; B 1.0732). But H25 shows fold-3 inner-CV noise of up to 0.050, larger than the gap that decided fold 3, so the noise clause refutes it.

- **시험한 원인**: C6/C7 at fold level: different evolved bases fit different folds. The question is whether choosing each fold's rule by that fold's inner-CV, from a pool fixed in advance, is a valid and reliable hyperparameter choice. This is standard nested CV, so the composite's holdout reads no holdout during selection. The test also checks how sensitive the procedure is to the candidate pool.
- **바꾼 것**: I made no new rule. For each fold I copied, verbatim, the per-fold rule (code, w0, extra features) of whichever earlier rule had the lowest inner-CV on that fold. Run A picks from the 8 selected non-control rules of rounds 1-3. Run B picks from every uncontaminated official result (32 files; the plan said 31, and the argmin is the same). Only fold 3 differs: it gets the plain c2 s3 base. The official runs check that each fold refits to exactly the same numbers, and they record the composite holdout.
- **돌린 것**: H28A_poolA: inner 1.0550 · 홀드아웃 1.0563 / H28B_poolB: inner 1.0528 · 홀드아웃 1.0732
- **판정**: refuted — Required disclosure: before these runs, the planner had already computed from existing per-fold RESULT values that A gives about 1.0563 holdout with 3 of 4 folds, and B about 1.0732 with 2 of 4. The official runs only check determinism and record the numbers.

**Confirm criteria: all met.**
- Determinism holds, and it is exact. In both runs, every fold matches its source's per-fold results with a difference of 0.0: train_gm, inner_cv_gm, holdout_gm, every fitted weight, every per-shape holdout regret and every pick. This holds even though six official runs shared the machine.
- One name collision needed checking. Fold 2 (H18A) and fold 1 (H22A) both use a feature named multistage_family_indicator, and the probe keeps the first definition it loads. The two definitions (stages>=3 and pipeline_kind=='multistage') agree on all 980,915 a6000 rows, so the values are the same, and the exact reproduction confirms it.
- Run A: holdout 1.0563, below the best single rule (1.0697) by 0.013 and below the vendor (1.0815) by 0.025. It beats the vendor in 3 of 4 folds (f0, f2, f3; f1 is 1.0977 against 1.0891).
- Run B: holdout 1.0732, below the vendor.

**Refute criteria**
- The reproduction clause and both holdout-threshold clauses do not fire.
- The H25 noise clause does fire. H25 refits H18A and H23B after only relabelling the weight indices, which leaves the objective identical. That alone moves per-fold inner-CV by:
  - H18A: +0.0039, +0.0350, -0.0041, +0.0050
  - H23B: -0.0033, +0.0096, +0.0097, -0.0504
- Fold 3: noise of up to 0.050 is larger than the fold-3 gap named in the clause (0.009). It is also larger than the gap that actually decided pool A's fold 3 (H19B 1.0784 against H01b 1.0817, 0.0033). So the fold-3 pick is set by fit-path noise.
- Fold 0: noise of 0.003-0.004 is well below the 0.021 gap, so the fold-0 choice of H23B is reliable.
- Two further gaps, outside the clause's wording, are also inside the noise: fold 1 (gap 0.027 against noise 0.035) and pool B's fold 2 (H18A against H23A, gap 0.00014).

Read literally, every number target is met, but the pre-registered reliability clause fires: 'the selection is not reliable even though it looks good'. So the verdict is refuted.

**What the noise does to the result.** These numbers are computed exactly from official per-fold values; they are not new runs.
- Swapping in the H25 relabelled refits of H18A and H23B flips fold 3's pick from H19B (holdout 1.021) to H23B' (1.045). The composite becomes 1.0625, still 3 of 4 folds.
- Pool B shows the same fragility with a real cost. Its fold 3 moves to the plain c2 s3 base on an inner-CV gap of 0.009, and that fold's holdout goes from 1.021 to 1.089, adding +0.0159 pooled log gap. Where that comes from:
  - the swizzle trap returns on 8192^3 (1.037 → 1.436) and 8192x12288x4096 (1.030 → 1.334);
  - 128x12288x4096 over-splits to sk8 (1.011 → 1.240).
  - So the wider pool loses 0.017, which is more than the ~0.01 run-to-run noise.

**Pool sensitivity** (exact, from per-fold RESULT values)
| Pool | Holdout | Folds beating vendor |
|---|---|---|
| c2 bases only | 1.1155 | 0/4 |
| r1-r2 selected rules | 1.0681 | 3/4 |
| all r1-r3 runs | 1.0616 | 3/4 |
| A leave-one-out (range) | 1.0562-1.0673 | 3/4 |
| A (official) | 1.0563 | 3/4 |
| B (official) | 1.0732 | 2/4 |

The level beats the vendor whenever the pool contains the C1/C3-fixed rules. Where it lands (1.056 to 1.073) depends on noisy picks.

**Per-class holdout, A (m1 classes)**
| Class | A | Vendor | Best single (H18A / H23B) |
|---|---|---|---|
| M<=32 | 1.012 | – | – |
| big | 1.051 | – | – |
| hungry | 1.056 | 1.069 | 1.059 / 1.061 |
| mid-easy | 1.105 | 1.102 | – |

Hungry is the best number recorded for that class. Pool B's big class is 1.120, because of the fold-3 trap.

**Caveats**
- The composite's inner-CV (1.0550 / 1.0528) is a minimum over candidates, so it is biased low. The chosen sources' own pooled inner-CVs are 1.072-1.088. Do not rank it against single rules.
- The pool is not blind to holdout at the research level. Round 2-3 rules were designed after reading earlier holdout results, and the pools were defined after the planner computed these composites. Nested CV removes the per-fold selection bias, but not this research-loop optimism.

**What this means for C6/C7.** Different bases really do win different folds on both inner-CV and holdout in folds 0 and 2. But per-fold inner-CV is too noisy to rank near-ties, because CMA's fit path alone moves it by up to 0.05. Per-fold picks are trustworthy only when the gap is well above about 0.01-0.05. This matches C7: single-run comparisons of fold-level inner-CV are unreliable.
- **상수의 출처**: No new constants. Each fold's code, w0 and extra_features are copied verbatim from the chosen source rule file's per_fold[f]; the only change is that relative extra_features paths are made absolute under /home/piai/workspace/kernelRule.

The source rules' constants come from their own hypotheses, all train-only:
- H23B: margin-formula b and c = 5.415 from fold-0 train.
- H22A: IQR-formula b_f from fold-1 train.
- H18A: band-P gate and per-fold swept b from fold-2 train.
- H19B: guard k_f and ramp c_f = 463052 from fold-3 train.
- c2_evolved_s3: the evolved base, fit on train only.

Selection used only per_fold[f].inner_cv_gm from existing official RESULT files, with argmin and ties broken by path. Pool A is the 8 inner-CV-selected non-control runs of rounds 1-3 (H02 excluded because its w0 is contaminated). Pool B is all 32 uncontaminated official 4-fold RESULT files under a6k/, excluding H02 A and B. Holdout was not read for selection. It was read only afterwards, for reporting and the post-hoc pool-sensitivity table.

## 반박 검증 (가장 좋은 둘)

### H28 · leakage — ⛔ 못 버팀

Nothing leaks mechanically: the reported 1.0563 does not come from any constant, axis or per-fold selection step reading holdout. The composites reproduce their source per-fold results exactly (diff 0.0). The four split-K gates and every numeric w0 constant re-derive exactly from train-only scripts, the extra axes read only shape, hw and config fields, and the per-fold choice reads only inner_cv_gm. The one constant not derived from train is the 2e5 large-grid gate, which the analyst admits picking after seeing all 65 shapes. It is membership-equivalent (0/65 flips) to each fold's train-derived split-K boundary, so it has no numeric effect. The headline number is still inflated at the selection level. The pool that gives 1.0563 leaves out the c2 evolved bases, and that was decided after the planner had computed both composites' holdouts. Adding c2_evolved_s3 alone gives 1.0732 and 2/4 folds. The fold-3 and fold-1 picks were near-ties inside fit-path noise, and both landed on the best fold holdout among the 8 candidates; the result is only 0.004 above the pool's holdout oracle. Noise-aware estimates are 1.061-1.073 (1.0673 for A with per-fold noise tolerance), still below vendor 1.0815 but about 0.01-0.017 worse than reported. So 1.0563 should not be cited as an unbiased estimate. This agrees with H28's 'refuted' verdict and adds that the headline number itself is optimistic. Audit scripts and logs: /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H28_leakage/ (v1_repro.py, v2_gates.py/.log, v3_luck.py/.log, v4_compare.py/.log, v5_prefilter.py/.log, v6_pools.py/.log, re_*/ re-derivations).

- The choice of candidate pool, made with holdout already known, sets the headline number. The whole gap between A (1.0563, 3/4 folds) and B (1.0732, 2/4 folds) comes from one inclusion: whether the four c2 evolved bases are in the pool. A plus only c2_evolved_s3 gives exactly 1.0732 and 2/4, because fold 3 flips from H19B (holdout 1.0212) to c2 s3 (1.0893). Adding the analyst probes to A changes nothing (1.0563). H28's own cause is 'different evolved bases fit different folds', so leaving the evolved bases out of a choose-a-base-per-fold pool is not principled. The disclosure says the planner computed both composites' holdout before the pools were fixed and the runs made.
- Near-tie luck inflates the number. The fold-3 pick (H19B over H01b, inner-CV gap 0.0033; H25 fit-path noise in that fold is up to 0.050) lands on the best fold-3 holdout of the 8 pool-A candidates. The fold-1 pick (H22A; noise 0.035, gap 0.027) is also the best holdout of 8. Pool A's holdout-oracle composite is 1.0522, so the reported 1.0563 is only 0.004 from the best the pool could give. If each fold instead takes the geometric mean of the candidates within that fold's observed noise, the expected composite is 1.0673 for pool A (1.0691 for pool B), still 3/4 folds. Dropping H19B gives 1.0609, and using the H25 relabelled refits gives 1.0625.
- One rule constant is not train-derived. The large-grid gate `output_work_per_sm > 200000` (folds 0-2, via r09/H01b/H20/H22) was, by the analyst's own note in analysis.json, 'picked after seeing all 65 shapes'. That breaks the letter of the honesty rule. It is numerically harmless, though. Among roofline>=1 shapes, 2e5 sits in the empty gap (199729, 268386), the same gap as each fold's train-derived split-K gate boundary (234057; 249661 for fold 2), so 0 of 65 shapes change membership if it is swapped for that train value. H21A/H21B, which replace it with a train-derived trap ramp, are worse on inner-CV too (pooled 1.0889/1.0848 against 1.0723/1.0771), so the gate is supported without holdout. A different train-only criterion, H19's trap c_f of 346k/463k/489k, would flip one holdout shape each in folds 1 and 2 (8192x4096x11008, 2048x11008x4096) and none in fold 0.
- Optimism at the research level, disclosed but real: the round 2-3 candidates were designed after reading earlier holdouts. single_resident_cta (fold 2, H18A) was motivated by the C4 analysis over all 65 shapes; its sign and scale are train-derived.
- The selection metric is itself biased toward heavily tuned rules. Gates (s19 balanced-accuracy search), L1 (b, c) grid picks, H01/H20 signs, H23 margin b_f and H18 sign were all chosen on the full fold train, and the full fold train contains the inner-CV validation parts. Inner-CV therefore flatters tuned rules. This does not leak into holdout, but it weakens the nested-CV argument. The composite's inner-CV (1.0550) is also a minimum over candidates, so it is biased low.
- No mechanical leakage was found. All 14 extra axes used read only p/hw/cfg fields (checked source). Every numeric constant re-derives exactly from train-only scripts. Selection (build.py select) reads only per_fold inner_cv_gm. The base w0 values are the c2 loop's w, fitted on the same nkband fold train (split nkband{f}-k4, 48/17; archive best chosen by train regret; no holdout in the LLM prompts). The multistage_family_indicator name collision has no effect (exact reproduction).

### H28 · stability — ⛔ 못 버팀

The perturbation: each fold's chosen component in H28A (pool A) was refit with only its new-term w0 scaled by x0.5 and x2. The code, extra features and base w0 were copied verbatim. Two official runs were made.

What stays robust: the composite always beats the c2 baselines by a wide margin. Holdout is 1.056-1.074 against s1's 1.1023, and inner-CV is 1.055-1.066 against s1's 1.0976. Fold 2 (H18A) returns exactly the same holdout, 1.049008, in all three fits. Pooled holdout also stays numerically below the vendor in both runs.

What does not hold is the size of the gain:
- **x0.5:** holdout 1.0589, 4/4 folds; it still holds.
- **x2:** holdout 1.0744, only 0.007 below the vendor, which is a tie under the 0.01 noise rule. That is worse than the best single rule (1.0697). The hard/hungry class (1.088) loses to the vendor (1.069), and one fold-1 shape jumps to regret 2.0.

Selection is also unstable: fold 3's H19B pick would switch to the runner-up H01b under both perturbations, because its inner-CV moves by up to 0.025. This matches the noise clause that refuted H28.

Three fits of the same composite spread 0.018 in holdout, more than the run-to-run noise. The reported 1.0563 (0.025 below the vendor, 0.013 below the best single rule) is therefore a favourable draw, not a stable level. holds_up=false.

Files, in /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H28_stability/:
- build.py, build.log, build_prov.json
- H28A_newterm_x0p5.json / .RESULT.json
- H28A_newterm_x2.json / .RESULT.json
- post.py, post.log, post.json

- Under the x2 perturbation the size of the gain does not hold. Holdout rises from 1.0563 to 1.0744, so the margin to the vendor (1.0815) shrinks from 0.025 to 0.0071. That is below the stated ~0.01 start-to-start noise, which makes it a tie. The composite also falls behind the best single rule (H23B, 1.0697), so under x2 per-fold selection no longer adds anything.
- The x2 run loses to the vendor on the class the method targets. Hard (sk>=3 optimum) gm is 1.08764 against the original 1.055573, and the hungry class is 1.088 against the vendor's 1.069. sk>=3 picks on hard shapes drop from 27/31 to 24/31.
- In fold 1 under x2, one shape fails badly. 2048x4096x11008 goes from regret 1.012 to 2.004 (pick 64x64x32 sk8, best 128x128x32 sk6), and fold-1 holdout moves from 1.0977 to 1.1348.
- The fold-3 pick is set by fit-path noise, as H28's own refutation said. Rerun with new-term w0 scaled, H19B's fold-3 inner-CV becomes 1.103778 (x0.5) or 1.084711 (x2), both worse than the unperturbed runner-up H01b (1.081701). The same selection rule would therefore switch fold 3 to H01b in both perturbations.
- Refitting the same four components three times gives holdouts of 1.0563, 1.0589 and 1.0744, a spread of 0.018, which is larger than the 0.01 noise. Per-fold inner-CV also moves by up to +0.025 (fold 3, x0.5) and holdout by up to +0.037 (fold 1, x2), consistent with the fit-path noise found in H25.
- Folds beating the vendor are not stable either: 4/4 at x0.5, 3/4 at x2. At x0.5, fold 1 beats the vendor by only 0.0028 (1.086262 against 1.089068), which is inside the noise.
- Limits of this check: only the chosen component in each fold was refit; the runners-up were not, so the re-selection comparisons use their original inner-CV. Only one perturbation family was tried (new-term w0 x0.5 and x2, where new terms are the weights beyond each fold's c2 evolved base: f0 w17-20, f1 w25-28, f2 w41-45, f3 w31-35).

### H28 · classes — 버팀

The class lens bears out H28's per-class figures (M<=32 1.012, big 1.051, hungry 1.056, mid-easy 1.105; pool B big 1.120; pool B fold-3 regressions 1.436, 1.334 and 1.240), and it does not change the refuted verdict. The verdict rests on inner-CV noise, which this lens does not test.

The improvement over the c2 s0 baseline is spread out, so holds_up is true. Pooled holdout goes from 1.1149 to 1.0563 (log −0.054). By class, hungry gives 58%, big 21%, mid-easy 18% and M<=32 3%. A is better than s0 on 39 of 65 shapes and worse on 11. It improves every fold. Dropping the 10 best shapes still leaves −0.017.

The margin over the vendor is thin and sits in one class. M<=32 supplies 86% of the −0.0236 log gap, and even s0 wins that class, because the vendor's nearest-mapped pick there is a 128x128 sk2 tile. Without M<=32, A is 1.0665 against the vendor's 1.0709, inside the noise. Dropping A's 10 best shapes against the vendor makes it a tie.

The composite's gain over the best single rules (−0.013 against H23B) is mostly mid-easy shapes from H22A in fold 1.

A still loses to the vendor on 18 shapes, for three reasons:
- **Swizzle trap (C3), the largest cause.** On 4096^3 A picks sk4s with identity-1 swizzle: 1.344, where the same split with identity-4/8 is 1.006. On 4096x4096x11008 it picks sk12s identity-1: 1.468, against 1.026 with identity-8.
- **Split-count errors (C1/C2).** 128x4096x11008 over-splits to sk8 (1.215). 512^3 picks sk4 parallel on a 128x128 tile (1.463). 128x11008x4096 in fold 2 under-splits to sk1 (1.226) although it sits in band P.
- **Big-tile cost in the big class (C4).** A picks 1-CTA 128x256 or 256x128 identity-1 tiles and loses 8 of 9 shapes by 3-10% each (class GM 1.051 against the vendor's 1.032). This is the price of avoiding 128x128 identity-1, which the rule cannot tell apart from the good swizzled 128x128 configs.

Scripts and logs, written only under /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H28_classes/: classes.py and classes.log (class and fold tables, rebuilt picks), classes.json, extra.py and extra.log (margin without M<=32, comparison with the single rules, fold-1 shapes), grid.py and grid.log (split-K by swizzle regret grids that attribute each loss). No official runs were made.

- The margin over the vendor sits almost entirely in the M<=32 class, and every rule wins that class. The pooled log gap A−vendor is −0.0236; M<=32 contributes −0.0202 of it (86%), hungry −0.0063, big +0.0024 (A loses 8 of 9 shapes), mid-easy +0.0005. The c2 s0 baseline also beats the vendor on M<=32 (1.0205 against 1.1295). There the vendor's nearest-mapped pick is 128x128x32 st6 sk2s even for M=1..32. Without M<=32, A is 1.0665 against the vendor's 1.0709 (a 0.004 gap, below the ~0.01 noise). Dropping A's 10 best shapes against the vendor leaves +0.0005, a tie.
- The composite's gain over the best single rules is concentrated in one class, mostly fold-1 shapes. Against H23B (1.0697 → 1.0563, log −0.0126), mid-easy gives −0.0087 (69%), from 128x4096x256, 1024x4096x256, 1024x4096x128 and 512x512x512, all in fold 1 (H22A). Against H18A (−0.0179), mid-easy gives −0.0152 (85%), led by 128x4096x4096 in fold 0 (1.672 → 1.047). So the part the per-fold selection adds is not spread; only the part over s0 is.
- Fold 1 (H22A) is the fold that loses to the vendor: 1.0977 against 1.0891. Mid-easy costs +0.0045 and hungry +0.0005; excluding M<=32, fold 1 is 1.1179 against 1.0872. Its three worst shapes are 512x512x512 at 1.463 (vendor 1.000), 4096x4096x11008 at 1.468 (1.054) and 128x4096x11008 at 1.215 (1.033), and together they cost more than fold 1's whole margin.
- The swizzle trap (C3) is still the main cause of the two biggest hungry-class losses. On 4096^3 (fold 0, H23B) A picks 128x128x32 st3 sk4s with identity-1 swizzle: 1.344. The same sk4 with identity-4/8 is 1.006, so all of this loss is swizzle, none of it split count. On 4096x4096x11008 (fold 1, H22A) A picks sk12s identity-1: 1.468. The same config with identity-8 is 1.026, while sk4s identity-1 is 1.291, so swizzle dominates and over-splitting adds a little.
- Split-count errors (C1/C2) remain on single shapes. 128x4096x11008 in fold 1 over-splits to sk8s (1.215; the optimum sk2s is 1.039). 128x11008x4096 in fold 2 (H18A) under-splits to sk1 (1.226; sk3s is 1.000; the vendor's sk6s is 1.060) even though the shape sits in band P (log_flops 33.4, output_work_per_sm 1.7e4). 512x512x512 (an 11 µs shape) picks 128x128x32 sk4 parallel (1.463), where the vendor's and the optimum's pick is 64x64x32 sk1.
- The big class (output_work_per_sm > 2e5) still loses to the vendor in 8 of 9 shapes: GM 1.0508 against 1.0324. A picks 1-CTA 128x256 or 256x128 st2/3 sk1 identity-1 tiles (1.030-1.101). The vendor picks 128x128x32 st5 identity-8 (1.03), and the optimum is 128x128x32 st3 identity-8, 2 CTA/SM. This is the cost of the tile route around C3: 128x128 with identity-1 is 1.284 on 8192x4096x4096, and the rule cannot see swizzle. So this is C4 at 3-10% per shape. Pool B's fold 3 (s3 base) loses this class outright: 1.1201, with 8192^3 at 1.436 and 8192x12288x4096 at 1.334.
- The class lens does not rescue the H28 verdict. The candidate's refutation rests on per-fold inner-CV noise (the fold-3 pick H19B against H01b, a 0.0033 gap). Fold 3's advantage over the vendor comes from hungry (−0.0056) and M<=32 (−0.0040), and it would change if that pick flipped.

### H23 · leakage — 버팀

I found no leak that inflates H23B's holdout of 1.0697, so the number holds up. I traced every constant in the rule:

- **Split-K gates:** found by a train-only search, which I re-ran under an audit that logs every read of measured times. All reads were train shapes, and the output was byte-identical to fold_gates.json.
- **Centre c and bowl strength b_f:** derive.py reads times only for the fold's train shapes (checked by the same audit) and rebuilds both rule files byte-identically. My own separate computation also gives the same c and b_f.
- **Large-grid branch weights:** set from train by r2_H20/derive.py, which the audit also passed and which reproduces H20A byte-identically.
- **Base code and starting weights:** the last train-selected entry of the c2 s1 loop. That loop fits on the same nkband fold train, and the LLM is sealed from validation.
- **Extra axes** (runs/c2-a6000-f*-s1/features.jsonl): read only p, hw and cfg, never times.

Between the two runs, B was picked by inner-CV (1.0775 against 1.0830). The problem is the large-grid threshold `output_work_per_sm > 200000`, inherited from r09. By the analyst's own note it was picked after seeing all 65 shapes. It changes no scores, though: each fold's own train threshold selects the same 9 shapes. Design choices informed by all 65 shapes and by earlier holdout reads add some optimism that I cannot measure. Files are in /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H23_leakage/: audit.py plus audit_h23/h20/s19/s19train.json; indep.py/indep.json (my recomputation); owps.py (shape axis listing); and rerun_derive/, rerun_h20/, rerun_s19/fold_gates_trainonly.json (reruns).

- The large-grid gate constant 200000 in `if p.roofline_ratio >= 1 and p.output_work_per_sm > 200000` came from r09 through H20A, and it did not come from train. The analyst's own record says so (analysis.json lenses[0].candidate_edits[0]: 'I picked it after seeing all 65 shapes, so it carries mild selection bias'). As written, that breaks the honesty rule. I checked that it has no effect on the result. Each fold's train-derived owps threshold (234057, 234057, 249661, 234057; the same values the s19 train-only gate search produced) puts exactly the same 9 of the 65 shapes in the branch (holdout counts 1/1/3/4). So on every shape the probe reads, the scores are identical, and the holdout number cannot be inflated by this constant. There is a knife-edge: 4096x4096x4096 and 4096x4096x11008 sit at owps 199728.8, just under 2e5. Both thresholds exclude them. In fold 2, holdout shape 2048x11008x4096 (268385.5) lies inside the train gap (199728.8 to 299593.1). Both 2e5 and the train midpoint 249661 include it.
- The rule's design was shaped by looking at all 65 shapes and at earlier official holdout reads (r09, L1, H18A, H20A): the interior-optimum quadratic form, the band-P gate idea, the existence and axes of the large-grid branch, and the choice of H20A as base. These are structural choices, not constants, and they are the same in all four folds. The optimism they add cannot be measured. H23 itself changes only two w0 entries per fold, through a pre-specified train-only formula. Its holdout (1.0697) ties its parent H20A (1.0710).
- Minor, and not a leak: the source gate script an/logs/s19_foldgates.py reads times for all 65 shapes and prints a 'holdout BA (reporting only)'. The gate search uses train labels only. I confirmed this with a train-only rewrite run under an audit: 0 reads of times outside train, and it produced a fold_gates.json byte-identical to the one used.

### H23 · stability — ⛔ 못 버팀

I tested the selected rule, H23B_margin_H20A (inner 1.0775, holdout 1.0697, 3/4 folds), by scaling its two split-K bowl weights in w0 by 0.5 and by 2 in every fold. The centre c and every other weight were left unchanged. No constants were re-derived, so nothing new was taken from holdout.

The rule's improvement over the c2 evolved baselines is robust. Both perturbed variants stay well ahead of s1 and s0 on inner and holdout, and still pick sk>=3 on most hard shapes.

The vendor win is fragile:
- x0.5 does as well or slightly better: holdout 1.0652, 3/4 folds.
- x2 falls to a tie with the vendor: holdout 1.0798 against 1.0815, and only 2/4 folds beaten. f3 turns into a loss through the large-grid swizzle trap (8192^3 goes from 1.047 to 1.436) and an over-split on 128x12288x4096.
- x2 has the best inner-CV of the three (1.0610), so selecting by inner-CV among these bowl strengths would pick the variant that only ties the vendor.
- The three variants differ by about 0.015 in holdout, with inner and holdout ranked in opposite orders. Fitted bowl strengths vary up to 5x between starts, and f1 swings by 0.05 on shapes outside the gate.

H23B's 0.012 margin under the vendor is therefore about one noise unit and depends on the start point. It is not a stable effect of the margin formula.

Files are in /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H23_stability/:
- build.py builds the perturbed rules.
- H23B_bowl_x0p5.json and H23B_bowl_x2.json are the rule files, with matching .check.log, .run.log and .RESULT.json for each.
- compare.py and compare.log hold the per-fold, per-class and per-shape comparison.

- The 0.012 margin by which H23B beats the vendor does not survive a factor-2 change in the bowl strength. With both bowl weights of w0 doubled in every fold (strength 2*b_f, same centre c), pooled holdout is 1.0798 against the vendor's 1.0815, a 0.0017 gap that is a tie at the stated 0.01 noise level. Folds beating the vendor drop from 3/4 to 2/4: f3 loses at 1.0777 against 1.0630 and f1 still loses at 1.1383. So the task's target (holdout < vendor AND more folds) is not met under x2.
- Following the selection protocol would pick the variant that ties the vendor. By inner-CV, x2 is the best of the three bowl strengths (1.0610, against 1.0732 for x0.5 and 1.0775 for H23B). By holdout it is the worst (1.0798, against 1.0652 and 1.0697). Across these perturbations the inner-CV and holdout rankings run in opposite directions, so the reported vendor win depends on which bowl strength was chosen, not only on the rule structure.
- x2 is a reasonable perturbation, not an extreme one. H23's own train-only diag.log puts the frozen-train best b at 2.1x, 2.9x and 5.3x b_f in folds f2, f0 and f3. Also, the IQR b used by H20A differs from b_f by 0.15x to 3.6x across folds.
- CMA does not pin down the fitted bowl. Fitted b is 25.4 / 5.2 / 9.1 in f2 and 1286 / 271 / 1183 in f3 (H23B / x0.5 / x2). f1 holdout swings between 1.145, 1.098 and 1.138, mostly on shapes outside the split-K gate (512x512x512 is picked with sk8 at 1.656 in H23B and at 1.091 in x0.5; 128x4096x256 comes out 1.364 / 1.091 / 1.455). Those are differences in the path CMA takes, not effects of the bowl.
- x2 loses f3 on the C3 large-grid swizzle trap: 8192x8192x8192 goes from 1.047 (128x256x32) to 1.436 (128x128x32, the tile with the best time, but the pick lands on a slow variant). 128x12288x4096 also over-splits (sk6 against best sk3, 1.011 to 1.154). Large-grid class gm is 1.095 for x2 against 1.060 for H23B and 1.032 for the vendor.
- What does hold: all three bowl strengths beat the c2 baselines clearly on both metrics. Inner is 1.061-1.078 against 1.0976 for s1 and 1.1177 for s0; holdout is 1.065-1.080 against 1.1023 for s1 and 1.1149 for s0. They also pick sk>=3 on 24-28 of 31 hard shapes, against 8 for s1. The improvement over the evolved baselines is robust; the vendor win is not.

### H23 · classes — 버팀

Run B (H23B, the rule selected by inner-CV) does improve on the s0 baseline broadly. The gain is -0.0414 log in total, over 36 shapes. Most of it is the hungry split-K class (-0.0291) and the large-grid class (-0.0101); M<=32 and mid are about flat. It is not carried by one or two shapes: the top 2 give 24% of the gain, and without them H23B is still 1.0700 against s0's 1.1051. So holds_up is true for the lens question.

Two caveats.
1. The win over the vendor (1.0697 against 1.0815) is at noise level. It is carried by the M<=32 class, which s0 already won, plus a small hungry gain. Large-grid and mid still lose to the vendor.
2. Against H20A the result is a tie that comes down to 3 shapes. Without its two biggest gains, H23B is worse (1.0681 against 1.0631).

The remaining vendor losses have three named causes:
- **f1 mid-class small-K / small-grid shapes** (512^3, 128x4096x256, 1024x4096x128, 1024x4096x256). The rule picks large 1-CTA tiles or a parallel split where the optimum is 64x64x32 with 3-4 CTA/SM. These shapes are outside the bowl gate, and they changed from H20A only because CMA refit the other weights differently.
- **Swizzle trap inside the gate.** 4096x4096x{4096, 11008} sit at output_work_per_sm 199728, just under the gates, so the bowl forces a serial split. The split level is right, but the tie resolves to swizzle 1: 1.344 and 1.394, where swz4 would give 1.006 and 1.009.
- **Large-grid tile family (C4).** The rule picks 128x256 1-CTA (1.04-1.10) where 128x128x32 st3 2-CTA would give 1.000-1.013 even at swizzle 1.

The overall H23 verdict (refuted by Run A) stands. Run A's hungry class is 1.102 against the vendor's 1.069.

Files, all in /tmp/claude-1000/-home-piai-workspace-kernelRule/d27f8d45-6378-4355-b573-0ac65dcd41f9/scratchpad/a6k/it/verify/H23_classes/:
- classes.py and classes.log: class, fold and baseline breakdown, with all 65 shapes and their pick, vendor pick and optimum configs
- family.py and family.log: split_k x swizzle regret grids for the worst losers
- per_shape.json

- The win over the vendor is thin and sits in the classes the base already won. Pooled log gap to the vendor is -0.0109 (1.0697 against 1.0815), which is about the noise level. By class: M<=32 -0.0198 (s0 alone already had -0.0187 here) and hungry -0.0040 help; large-grid +0.0037 and mid +0.0092 still lose to the vendor. The top 5 shapes give 106% of the net vendor margin.
- Fold 1 still loses to the vendor, 1.145 against 1.089. Most of the loss is the mid class: 5 shapes at gm 1.312 against the vendor's 1.098. 512x512x512 scores 1.656: it picks 128x128x32 sk8 parallel, while both the vendor and the optimum use 64x64x32 sk1 (3 CTA). 128x4096x256 scores 1.364 (picks 128x128x32 1-CTA; best is 64x64x32 4-CTA). 1024x4096x128 scores 1.269 (picks 128x256x32; best 64x64x32 4-CTA). 1024x4096x256 scores 1.216 (picks 128x256x32; best 128x128x32 2-CTA). Every one of these is outside the split-K gate (log_flops 28-30), so the bowl cannot be the cause.
- The swizzle trap (C3) fires inside the bowl gate. 4096x4096x11008 (f1) scores 1.394 against the vendor's 1.054, and 4096x4096x4096 (f0) scores 1.344 against 1.054. Their output_work_per_sm of 199728 is just under the 2e5/234057 gates, so the bowl pushes them into serial split (sk6/sk4). The split level is nearly right, but the rule cannot see swizzle and ties resolve to width 1. The same 128x128x32 st3 config with swz4 would score 1.009 and 1.006. These two shapes add +0.0080 to the pooled log gap.
- Large-grid class (1.060 against the vendor's 1.032) is a tile-family error (C4), not swizzle. The rule picks 128x256 or 256x128 1-CTA sk1 and scores 1.038-1.096. The same shapes with 128x128x32 st3 2-CTA and serial sk1-2, even at swizzle 1, score 1.000-1.013: 2048x12288x4096 at sk2 swz1 is 1.000 against the pick's 1.096, and 2048x11008x4096 at sk2 swz1 is 1.002 against 1.090.
- The bowl is too weak or wrong-tiled on some hungry shapes. 2048x2048x2048 (f3) is inside the gate but picks sk1 and scores 1.101; sk3 serial scores 1.000, and the vendor's sk3 scores 1.005. 3000x4096x4096 (f0) picks 64x256x32 st2 sk4 and scores 1.219, against the optimum 128x128x32 sk3; the same family at sk4 swz1 would score 1.051.
- Against H20A (its direct IQR-b predecessor) the result is a tie, net -0.0012 log, and the difference comes from 3 shapes. It gains on f2 8192x11008x4096 (1.328 to 1.030) and loses on f1 128x4096x256 (1.091 to 1.364) and 1024x4096x256 (1.027 to 1.216). Without the two biggest-gain shapes, H23B is worse: 1.0681 against 1.0631. The f1 losses are outside the gate. They come from CMA landing somewhere else on the non-bowl weights (31 of 33 f1 weights differ by more than 20% from H20A's), which is fit variance, not the margin formula.
- The H23 verdict (refuted) still stands, and the class breakdown supports it. Run A's hungry class is 1.102 against the vendor's 1.069 (+0.0141). On H18A's base the H18A-to-Run A move has the largest per-shape regret changes on the fold 0 4096-K hungry shapes (for example 384x4096x4096 goes from 1.074 to 1.507).

