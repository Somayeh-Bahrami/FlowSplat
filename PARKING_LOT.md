# Idea Parking Lot

Do not open before Nov 16, 2026. New ideas go here, one line each, and are
not worked on until then (plan, Focus Rules).

| Date | Idea | Why it came up |
|---|---|---|
| Sep 29 | Cardiac motion + contrast filling disentanglement | Needed for real coronary cine (4DRGS limitation) |
| Sep 29 | TreeSplat: topology-native graph Gaussians for full trees | Connectivity problem from previous project |
| Oct 5 | Projection-space radius refinement (Xie et al., EMBC 2010) for poorly resolved throats | External feedback on Step 2.1 (122_0 throat) |
| Oct 5 | Check whether the 4DRGS radius bias comes from the reconstruction or from the half-max rule (cross-section profiles) | Step 2.2 run 5: bias grows with vessel size |
| Oct 6 | Estimate the lumen area per case from the projections instead of one global factor k | Step 2.2 run 6: per-case area bias spread limits Q |
| Oct 7 | Examine 131_1 as a failure case (throat mis-sized at every view count) | Step 3.1 views sweep |
| Oct 7 | Recalibrate k per noise level or estimate it from the data | Step 3.1 noise high: Q 20.9 % |
| Oct 7 | Lesion finder: minimum length or contrast check against false narrowings | Step 3.1: 3 misplaced lesions in about 230 runs |
| Oct 8 | Hard-regime comparison FlowSplat vs B1 (30 views, fast flow, high noise) on the test split | Step 4.1: B1 within 1 point of FlowSplat at 60 views (p = 0.52); plan fallback content, Gate 2 decision |
| Oct 9 | Tree extraction for short parents (e.g. lower threshold near the ostium, or seed the parent from the inlet end) | Step 5.1: 5 of 40 Ys failed, all with parents < 25 mm |
| Oct 9 | Murray prior from radii measured away from the junction | Step 5.1: prior with junction radii did not help |
