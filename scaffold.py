"""
Sutton and Barto from Scratch 5: Mars Lander from Flight Logs scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sutton & Barto from scratch 5: Mars Lander from Flight Logs.
Feature construction and LSTD (chapter 9), off-policy divergence and its fixes (chapter 11), true online TD(lambda) (chapter 12)."""
import itertools
import numpy as np


def sqrt_ve(X, w, truth):
    return float(np.sqrt(np.mean((X @ w - truth) ** 2)))


def main():
    # --- Part 1: the lander, its two autopilots, the flight logs and the exact values ---
    logs = fly("cautious", 500, np.random.default_rng(1))
    D = logs["dones"]
    print("Flight logs from the cautious autopilot:", int(D.sum()), "flights,", len(D), "transitions")
    print("Landed safely in the logs:", round(float(logs["rewards"][D].mean()), 3))
    pi_logs = fly("fast", 500, np.random.default_rng(4))
    fuel_b = float(logs["next_states"][D][:, 2].mean())
    fuel_pi = float(pi_logs["next_states"][pi_logs["dones"]][:, 2].mean())
    print("Mean fuel left at touchdown: cautious", round(fuel_b, 1), "| fast", round(fuel_pi, 1))
    V_b, V_pi = exact_values("cautious"), exact_values("fast")
    h0, v0, f0 = START
    print("Exact landing probability from the start (dynamic programming): cautious",
          round(float(V_b[h0, v0 + MAX_SPEED, f0]), 4), "| fast", round(float(V_pi[h0, v0 + MAX_SPEED, f0]), 4))
    S = logs["states"]
    truth = V_b[S[:, 0], S[:, 1] + MAX_SPEED, S[:, 2]]

    # --- Part 2: features, scored by sqrt(VE) of their LSTD solution over the logged states ---
    raw = polynomial_features(S)
    scale = np.sqrt((raw ** 2).mean(axis=0))
    grid = np.array(list(itertools.product(np.linspace(0, 1, 4), repeat=3)))
    families = {
        "polynomial, order 1 (raw telemetry / RMS)": lambda s: polynomial_features(s, 1, scale),
        "Fourier, order 2": lambda s: fourier_features(s, 2),
        "Fourier, order 3": lambda s: fourier_features(s, 3),
        "radial basis, 4x4x4 grid, sigma 0.2": lambda s: rbf_features(s, grid, 0.2),
        "tile coding, 4 tilings of 3x3x3": lambda s: tile_features(s, 4, 3),
    }
    print("On-policy prediction of the cautious autopilot (LSTD on all 500 flights):")
    for name, feat in families.items():
        X = feat(S)
        w = lstd(X, feat(logs["next_states"]), logs["rewards"], D)
        print(f"  {name:42s} d={X.shape[1]:3d}  alpha rule (tau=10)={step_size_rule(X, 10):.4f}  sqrt(VE)={sqrt_ve(X, w, truth):.4f}")
    sample = np.arange(0, len(S), 20)
    kv = kernel_value(S[sample], S, logs["returns"], 0.06)
    print(f"  {'kernel regression on returns, bandwidth 0.06':42s}                              sqrt(VE)={float(np.sqrt(np.mean((kv - truth[sample]) ** 2))):.4f}")

    # --- Part 3: data efficiency (LSTD vs semi-gradient TD) and interest/emphasis (Example 9.4) ---
    feat = families["Fourier, order 3"]
    X_all = feat(S)
    for k in (10, 50, 500):
        m = logs["flight"] < k
        X, X_next = feat(S[m]), feat(logs["next_states"][m])
        w_l = lstd(X, X_next, logs["rewards"][m], D[m])
        w_t = semi_gradient_td(X, X_next, logs["rewards"][m], D[m], step_size_rule(X, 10))
        print(f"After {k:3d} flights: LSTD sqrt(VE)={sqrt_ve(X_all, w_l, truth):.4f}   semi-gradient TD(0) sqrt(VE)={sqrt_ve(X_all, w_t, truth):.4f}")
    print("Example 9.4, two-step TD: without emphasis", np.round(interest_emphasis_td(2, 0.05, 2000, False), 3).tolist(),
          "| with emphasis", np.round(interest_emphasis_td(2, 0.05, 2000, True), 3).tolist())

    # --- Part 4: the incident. Evaluate the fast autopilot from the cautious logs ---
    rho, rho_to_end = importance_ratios(logs)
    print("Importance ratios pi/b on the logs: max", round(float(rho.max()), 2), "| mean", round(float(rho.mean()), 3),
          "| share below 0.1:", round(float((rho < 0.1).mean()), 3))
    poly = families["polynomial, order 1 (raw telemetry / RMS)"]
    X, X_next = poly(S), poly(logs["next_states"])
    alpha = step_size_rule(X * np.sqrt(rho)[:, None], 1)
    w_semi, norms = expected_td_sweeps(X, X_next, logs["rewards"], D, alpha, 600, rho=rho)
    print("Off-policy semi-gradient TD on the logs, |w| after sweeps 1, 150, 300, 450, 600:",
          [f"{norms[i]:.2e}" for i in (0, 149, 299, 449, 599)])
    post = deadly_triad_postmortem(logs, pi_logs, poly, lambda s: tile_features(s, 1, 3), 1, 600)
    print("Post-mortem, |w| after 600 sweeps:")
    for name, norm in post.items():
        print(f"  {name:22s} {norm:.3e}")
    print("Tsitsiklis-Van Roy, least-squares DP w_k (k = 0..5): gamma 0.9", np.round(tsitsiklis_van_roy(1.0, 0.9, 0.0, 5), 3).tolist(),
          "| gamma 0.8", np.round(tsitsiklis_van_roy(1.0, 0.8, 0.0, 5), 3).tolist())
    td_hist = baird_counterexample(1000, 0.01, np.random.default_rng(6))
    dp_hist = baird_counterexample(1000, 0.01)
    print("Baird, semi-gradient off-policy TD: w8 at steps 0, 500, 1000 =", np.round(td_hist[[0, 500, 1000], 7], 1).tolist())
    print("Baird, semi-gradient DP (expected update): w8 at sweeps 0, 500, 1000 =", np.round(dp_hist[[0, 500, 1000], 7], 1).tolist())

    # --- Part 5: the geometry of the objectives, the fixes, and the go/no-go call ---
    Xs = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P = np.zeros((4, 5)); P[0, 2] = P[1, 3] = P[2, 4] = P[3, 4] = 1.0
    R = np.zeros((4, 5)); R[2, 4] = 1.0
    mu, v_true = np.full(4, 0.25), np.array([1.0, 0.0, 1.0, 0.0])
    w_td, w_be = td_and_be_solutions(Xs, mu, P, R)
    print("A-presplit: TD fixed point (A, B, C) =", (np.round(w_td, 3) + 0.0).tolist(), "| BE minimiser =", (np.round(w_be, 3) + 0.0).tolist())
    for label, wv in (("TD fixed point", w_td), ("BE minimiser", w_be)):
        obj = value_error_objectives(Xs, wv, mu, P, R, v_true)
        print(f"  {label:15s} VE={obj['VE']:.4f} BE={obj['BE']:.4f} PBE={obj['PBE']:.4f} TDE={obj['TDE']:.4f}")
    Xb, w0 = baird_features(), np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
    Pb = np.zeros((7, 8)); Pb[:, 6] = 1.0
    mub = np.full(7, 1 / 7)
    expected_args = (Xb, np.tile(Xb[6], (7, 1)), np.zeros(7), np.zeros(7, dtype=bool))
    w_tdc_b, _ = expected_tdc_sweeps(*expected_args, np.ones(7), 0.005, 0.05, 1000, gamma=0.99, w0=w0)
    o = value_error_objectives(Xb, w_tdc_b, mub, Pb, np.zeros((7, 8)), np.zeros(7), gamma=0.99)
    print(f"Baird, expected TDC after 1000 sweeps: sqrt(PBE)={np.sqrt(o['PBE']):.4f}  sqrt(VE)={np.sqrt(o['VE']):.3f}")
    m = np.linalg.solve(np.eye(7) - 0.99 * Pb[:, :7].T, mub) * 7
    w_em_b, _ = expected_td_sweeps(*expected_args, 0.03, 1000, emphasis=m, gamma=0.99, w0=w0)
    o = value_error_objectives(Xb, w_em_b, mub, Pb, np.zeros((7, 8)), np.zeros(7), gamma=0.99)
    print(f"Baird, expected emphatic TD after 1000 sweeps: sqrt(VE)={np.sqrt(o['VE']):.4f}")

    x0 = poly([START])[0]
    estimates = {"semi-gradient TD": float(x0 @ w_semi)}
    agg = lambda s: tile_features(s, 1, 3)
    estimates["state aggregation (LSTD)"] = float(agg([START])[0] @ lstd(agg(S), agg(logs["next_states"]), logs["rewards"], D, rho=rho))
    starts = np.flatnonzero(np.r_[True, D[:-1]])
    weights = rho_to_end[starts]
    estimates["Monte Carlo, ordinary IS"] = float(np.mean(weights * logs["returns"][starts]))
    estimates["Monte Carlo, weighted IS"] = float(np.sum(weights * logs["returns"][starts]) / np.sum(weights))
    estimates["off-policy LSTD"] = float(x0 @ lstd(X, X_next, logs["rewards"], D, rho=rho))
    w_tdc, _ = expected_tdc_sweeps(X, X_next, logs["rewards"], D, rho, 0.1, 0.1, 10000)
    estimates["TDC, 10000 sweeps"] = float(x0 @ w_tdc)
    w_gtd2, _ = expected_tdc_sweeps(X, X_next, logs["rewards"], D, rho, 0.1, 0.1, 10000, gtd2=True)
    estimates["GTD2, 10000 sweeps"] = float(x0 @ w_gtd2)
    M = follow_on_emphasis(rho, D)
    w_em, _ = expected_td_sweeps(X, X_next, logs["rewards"], D, step_size_rule(X * np.sqrt(M * rho)[:, None], 1), 3000,
                                 rho=rho, emphasis=M)
    estimates["emphatic TD"] = float(x0 @ w_em)
    truth_pi = landing_rate("fast", 4000, np.random.default_rng(3))
    print("Go/no-go: switch to the fast autopilot only if its landing rate is at least 0.90")
    for name, value, call, ok in go_no_go_table(estimates, truth_pi, 0.90):
        shown = "     --" if value is None else f"{value:7.4f}"
        print(f"  {name:26s} {shown}  {call:8s} {'agrees with the truth' if ok else 'WRONG'}")
    print(f"  {'truth: fly it 4000 times':26s} {truth_pi:7.4f}  {'GO' if truth_pi >= 0.9 else 'NO-GO'}"
          f"   (exact value {float(V_pi[h0, v0 + MAX_SPEED, f0]):.4f})")

    # --- true online TD(lambda) vs accumulating TD(lambda), on-policy on the cautious logs (figure 12.8's lesson) ---
    tiles = lambda s: tile_features(s, 8, 4)
    Xt, Xt_next = tiles(S), tiles(logs["next_states"])
    for a in (0.5, 1.0, 1.5):
        row = []
        for true_online in (False, True):
            wl = td_lambda(Xt, Xt_next, logs["rewards"], D, a / 8, 0.9, true_online)
            err = sqrt_ve(Xt, wl, truth)
            row.append("diverged" if not np.isfinite(err) or err > 10 else f"{err:.4f}")
        print(f"TD(lambda=0.9) with alpha*|x|^2={a}: accumulating sqrt(VE)={row[0]:>8s}   true online sqrt(VE)={row[1]}")


if __name__ == "__main__":
    main()

