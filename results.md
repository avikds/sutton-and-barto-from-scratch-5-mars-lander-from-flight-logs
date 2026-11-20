```
Flight logs from the cautious autopilot: 500 flights, 8384 transitions
Landed safely in the logs: 0.924
Mean fuel left at touchdown: cautious 8.7 | fast 15.5
Exact landing probability from the start (dynamic programming): cautious 0.9305 | fast 0.8765
On-policy prediction of the cautious autopilot (LSTD on all 500 flights):
  polynomial, order 1 (raw telemetry / RMS)  d=  8  alpha rule (tau=10)=0.0125  sqrt(VE)=0.0589
  Fourier, order 2                           d= 27  alpha rule (tau=10)=0.0072  sqrt(VE)=0.0461
  Fourier, order 3                           d= 64  alpha rule (tau=10)=0.0031  sqrt(VE)=0.0394
  radial basis, 4x4x4 grid, sigma 0.2        d= 64  alpha rule (tau=10)=0.0838  sqrt(VE)=0.0403
  tile coding, 4 tilings of 3x3x3            d=256  alpha rule (tau=10)=0.0250  sqrt(VE)=0.0456
  kernel regression on returns, bandwidth 0.06                              sqrt(VE)=0.0412
After  10 flights: LSTD sqrt(VE)=0.3514   semi-gradient TD(0) sqrt(VE)=0.8662
After  50 flights: LSTD sqrt(VE)=0.1161   semi-gradient TD(0) sqrt(VE)=0.5480
After 500 flights: LSTD sqrt(VE)=0.0394   semi-gradient TD(0) sqrt(VE)=0.0682
Example 9.4, two-step TD: without emphasis [3.487, 1.487] | with emphasis [4.0, 2.0]
Importance ratios pi/b on the logs: max 13.5 | mean 0.985 | share below 0.1: 0.319
Off-policy semi-gradient TD on the logs, |w| after sweeps 1, 150, 300, 450, 600: ['1.17e-02', '8.00e+00', '3.91e+02', '1.83e+04', '8.44e+05']
Post-mortem, |w| after 600 sweeps:
  all three legs         8.441e+05
  on-policy data         6.837e-01
  state aggregation      3.305e+00
  Monte Carlo targets    8.780e-01
Tsitsiklis-Van Roy, least-squares DP w_k (k = 0..5): gamma 0.9 [1.0, 1.08, 1.166, 1.26, 1.36, 1.469] | gamma 0.8 [1.0, 0.96, 0.922, 0.885, 0.849, 0.815]
Baird, semi-gradient off-policy TD: w8 at steps 0, 500, 1000 = [1.0, 84.0, 325.4]
Baird, semi-gradient DP (expected update): w8 at sweeps 0, 500, 1000 = [1.0, 76.8, 335.0]
A-presplit: TD fixed point (A, B, C) = [0.5, 1.0, 0.0] | BE minimiser = [0.5, 0.75, 0.25]
  TD fixed point  VE=0.1250 BE=0.1250 PBE=0.0000 TDE=0.1250
  BE minimiser    VE=0.1562 BE=0.0625 PBE=0.0312 TDE=0.0625
Baird, expected TDC after 1000 sweeps: sqrt(PBE)=0.0074  sqrt(VE)=1.935
Baird, expected emphatic TD after 1000 sweeps: sqrt(VE)=0.0000
Go/no-go: switch to the fast autopilot only if its landing rate is at least 0.90
  semi-gradient TD                --  DIVERGED WRONG
  state aggregation (LSTD)    0.9336  GO       WRONG
  Monte Carlo, ordinary IS    0.6951  NO-GO    agrees with the truth
  Monte Carlo, weighted IS    0.9660  GO       WRONG
  off-policy LSTD             0.8126  NO-GO    agrees with the truth
  TDC, 10000 sweeps           0.8247  NO-GO    agrees with the truth
  GTD2, 10000 sweeps          0.8131  NO-GO    agrees with the truth
  emphatic TD                 0.9267  GO       WRONG
  truth: fly it 4000 times    0.8795  NO-GO   (exact value 0.8765)
TD(lambda=0.9) with alpha*|x|^2=0.5: accumulating sqrt(VE)=  0.0871   true online sqrt(VE)=0.0802
TD(lambda=0.9) with alpha*|x|^2=1.0: accumulating sqrt(VE)=  0.2546   true online sqrt(VE)=0.0884
TD(lambda=0.9) with alpha*|x|^2=1.5: accumulating sqrt(VE)=diverged   true online sqrt(VE)=0.1016
```
