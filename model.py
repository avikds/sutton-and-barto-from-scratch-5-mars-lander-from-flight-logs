"""
Sutton and Barto from Scratch 5: Mars Lander from Flight Logs

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - lander_step
import numpy as np

MAX_ALTITUDE = 16
MAX_SPEED = 5
MAX_FUEL = 24
SAFE_SPEED = 2
START = (16, 0, 24)


def lander_step(state, action, rng):
    """
    Advance the Mars lander by one second.

    Parameters:
        state: Tuple (altitude, vertical_velocity, fuel).
        action: Thrust level (0, 1, or 2).
        rng: Random number generator with a random() method.

    Returns:
        A tuple containing (next_state, reward, done).
    """

    # Unpack the current state.
    h, v, f = state

    # Apply thrust, limited by the amount of remaining fuel.
    thrust = min(int(action), int(f))
    new_fuel = int(f - thrust)

    # Draw exactly one random gust.
    u = rng.random()

    if u < 0.1:
        gust = -1
    elif u >= 0.9:
        gust = 1
    else:
        gust = 0

    # Update vertical velocity:
    # Gravity reduces velocity by 1 m/s every second.
    new_velocity = int(np.clip(
        v + thrust - 1 + gust,
        -MAX_SPEED,
        MAX_SPEED
    ))

    # Update altitude using the new vertical velocity.
    new_altitude = int(h + new_velocity)

    # Check whether the lander has touched down.
    if new_altitude <= 0:

        # A landing is successful if the downward speed
        # does not exceed the permitted safe landing speed.
        reward = 1.0 if new_velocity >= -SAFE_SPEED else 0.0

        # Set altitude to zero to represent touchdown.
        next_state = (0, int(new_velocity), int(new_fuel))

        return next_state, reward, True

    # Otherwise, the lander remains airborne.
    next_state = (
        int(min(new_altitude, MAX_ALTITUDE)),
        int(new_velocity),
        int(new_fuel)
    )

    return next_state, 0.0, False

# Step 2 - autopilot_probs (not yet solved)
# TODO: implement

# Step 3 - fly (not yet solved)
# TODO: implement

# Step 4 - landing_rate (not yet solved)
# TODO: implement

# Step 5 - exact_values (not yet solved)
# TODO: implement

# Step 6 - polynomial_features (not yet solved)
# TODO: implement

# Step 7 - fourier_features (not yet solved)
# TODO: implement

# Step 8 - rbf_features (not yet solved)
# TODO: implement

# Step 9 - tile_features (not yet solved)
# TODO: implement

# Step 10 - step_size_rule (not yet solved)
# TODO: implement

# Step 11 - semi_gradient_td (not yet solved)
# TODO: implement

# Step 12 - lstd (not yet solved)
# TODO: implement

# Step 13 - kernel_value (not yet solved)
# TODO: implement

# Step 14 - interest_emphasis_td (not yet solved)
# TODO: implement

# Step 15 - importance_ratios (not yet solved)
# TODO: implement

# Step 16 - expected_td_sweeps (not yet solved)
# TODO: implement

# Step 17 - deadly_triad_postmortem (not yet solved)
# TODO: implement

# Step 18 - tsitsiklis_van_roy (not yet solved)
# TODO: implement

# Step 19 - baird_counterexample (not yet solved)
# TODO: implement

# Step 20 - value_error_objectives (not yet solved)
# TODO: implement

# Step 21 - td_and_be_solutions (not yet solved)
# TODO: implement

# Step 22 - expected_tdc_sweeps (not yet solved)
# TODO: implement

# Step 23 - follow_on_emphasis (not yet solved)
# TODO: implement

# Step 24 - td_lambda (not yet solved)
# TODO: implement

# Step 25 - go_no_go_table (not yet solved)
# TODO: implement

