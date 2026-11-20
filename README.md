# Sutton and Barto from Scratch 5: Mars Lander from Flight Logs

Mission control wants a faster, fuel-saving autopilot for a Mars lander, but the only data is 500 flight logs from the cautious autopilot already in service. Build the lander and its exact values by dynamic programming, construct polynomial, Fourier, radial-basis and tile-coding features, and learn the cautious autopilot's landing probability with semi-gradient TD, LSTD and kernel regression. Then evaluate the new autopilot from the old logs: watch off-policy semi-gradient TD diverge, take the deadly triad apart one leg at a time, reproduce the Tsitsiklis-Van Roy and Baird counterexamples, compare VE, BE, PBE and TDE, and fix the problem with TDC, GTD2 and emphatic TD before making the go/no-go call.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** lander_step
- [x] **2.** autopilot_probs
- [x] **3.** fly
- [x] **4.** landing_rate
- [x] **5.** exact_values
- [x] **6.** polynomial_features
- [x] **7.** fourier_features
- [x] **8.** rbf_features
- [x] **9.** tile_features
- [x] **10.** step_size_rule
- [x] **11.** semi_gradient_td
- [x] **12.** lstd
- [x] **13.** kernel_value
- [x] **14.** interest_emphasis_td
- [x] **15.** importance_ratios
- [x] **16.** expected_td_sweeps
- [x] **17.** deadly_triad_postmortem
- [x] **18.** tsitsiklis_van_roy
- [x] **19.** baird_counterexample
- [x] **20.** value_error_objectives
- [x] **21.** td_and_be_solutions
- [x] **22.** expected_tdc_sweeps
- [x] **23.** follow_on_emphasis
- [ ] **24.** td_lambda
- [ ] **25.** go_no_go_table

---

Built on Deep-ML.
