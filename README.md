# Sutton and Barto from Scratch 5: Mars Lander from Flight Logs

Mission control wants a faster, fuel-saving autopilot for a Mars lander, but the only data is 500 flight logs from the cautious autopilot already in service. Build the lander and its exact values by dynamic programming, construct polynomial, Fourier, radial-basis and tile-coding features, and learn the cautious autopilot's landing probability with semi-gradient TD, LSTD and kernel regression. Then evaluate the new autopilot from the old logs: watch off-policy semi-gradient TD diverge, take the deadly triad apart one leg at a time, reproduce the Tsitsiklis-Van Roy and Baird counterexamples, compare VE, BE, PBE and TDE, and fix the problem with TDC, GTD2 and emphatic TD before making the go/no-go call.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** lander_step
- [ ] **2.** autopilot_probs
- [ ] **3.** fly
- [ ] **4.** landing_rate
- [ ] **5.** exact_values
- [ ] **6.** polynomial_features
- [ ] **7.** fourier_features
- [ ] **8.** rbf_features
- [ ] **9.** tile_features
- [ ] **10.** step_size_rule
- [ ] **11.** semi_gradient_td
- [ ] **12.** lstd
- [ ] **13.** kernel_value
- [ ] **14.** interest_emphasis_td
- [ ] **15.** importance_ratios
- [ ] **16.** expected_td_sweeps
- [ ] **17.** deadly_triad_postmortem
- [ ] **18.** tsitsiklis_van_roy
- [ ] **19.** baird_counterexample
- [ ] **20.** value_error_objectives
- [ ] **21.** td_and_be_solutions
- [ ] **22.** expected_tdc_sweeps
- [ ] **23.** follow_on_emphasis
- [ ] **24.** td_lambda
- [ ] **25.** go_no_go_table

---

Built on Deep-ML.
