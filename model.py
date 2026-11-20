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

# Step 2 - autopilot_probs
def autopilot_probs(state, kind):
    """
    Return the action probabilities of the specified autopilot.

    Parameters:
        state: Tuple (altitude, vertical_velocity, fuel).
        kind: Either 'cautious' or 'fast'.

    Returns:
        A float NumPy array of shape (3,) containing
        probabilities for actions 0, 1, and 2.
    """

    h, v, f = state

    # Define the target velocity and exploration probability
    # for the selected autopilot.
    if kind == "cautious":
        target = -1
        eps = 0.2

    elif kind == "fast":
        target = -3 if h >= 9 else -1
        eps = 0.15

    else:
        raise ValueError("Invalid autopilot kind. Use 'cautious' or 'fast'.")

    # Determine the rule-based thrust action.
    # Action 2: Increase thrust if falling faster than the target.
    # Action 1: Hold thrust if the velocity matches the target.
    # Action 0: Reduce fuel consumption if falling slower than the target.
    if v < target:
        rule_action = 2
    elif v == target:
        rule_action = 1
    else:
        rule_action = 0

    # Assign equal exploration probability to all three actions.
    probs = np.full(3, eps / 3, dtype=float)

    # Add the remaining probability to the rule-based action.
    probs[rule_action] += 1.0 - eps

    return probs

# Step 3 - fly
def fly(kind, n_flights, rng, start=START):
    """
    Simulate complete Mars landing flights and record their transitions.

    Parameters:
        kind: Autopilot type ('cautious' or 'fast').
        n_flights: Number of flights to simulate.
        rng: Random number generator.
        start: Initial lander state. Defaults to START.

    Returns:
        A dictionary containing NumPy arrays for:
        states, actions, rewards, next_states, dones,
        flight indices, and discounted returns (gamma = 1).
    """

    # Store transitions from all flights in chronological order.
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    flights = []

    # Simulate the requested number of complete flights.
    for flight_idx in range(n_flights):

        # Each flight starts from the specified initial state.
        state = tuple(int(x) for x in start)

        # Continue until the lander touches down.
        while True:

            # Record the current state before taking an action.
            states.append(state)

            # Obtain the autopilot's action probabilities.
            probs = autopilot_probs(state, kind)

            # Sample an action using the inverse cumulative distribution.
            # The action is always between 0 and 2.
            action = min(
                int(np.searchsorted(
                    np.cumsum(probs),
                    rng.random(),
                    side="right"
                )),
                2
            )

            # Advance the lander by one second.
            # lander_step draws its own random gust.
            next_state, reward, done = lander_step(
                state, action, rng
            )

            # Record the transition.
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(next_state)
            dones.append(bool(done))
            flights.append(int(flight_idx))

            # Move to the next state.
            state = next_state

            # End this flight immediately after touchdown.
            if done:
                break

    # Convert the recorded transitions into NumPy arrays.
    states = np.asarray(states, dtype=int).reshape(-1, 3)
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    next_states = np.asarray(next_states, dtype=int).reshape(-1, 3)
    dones = np.asarray(dones, dtype=bool)
    flights = np.asarray(flights, dtype=int)

    # Compute the undiscounted return (gamma = 1) for every transition.
    # Returns are calculated backwards and reset at flight boundaries.
    returns = np.zeros(len(rewards), dtype=float)
    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):

        # Reset the cumulative return when entering a new flight
        # during the backward pass.
        if t == len(rewards) - 1 or flights[t] != flights[t + 1]:
            G = 0.0

        G = rewards[t] + G
        returns[t] = G

    # Return the complete flight-log dataset.
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "flight": flights,
        "returns": returns
    }

# Step 4 - landing_rate
def landing_rate(kind, n_flights, rng, start=START):
    """
    Estimate an autopilot's probability of landing safely.

    Parameters:
        kind: Autopilot type ('cautious' or 'fast').
        n_flights: Number of flights to simulate.
        rng: Random number generator.
        start: Initial lander state. Defaults to START.

    Returns:
        Python float representing the fraction of safe landings.
    """

    # Simulate the specified number of complete flights.
    logs = fly(kind, n_flights, rng, start)

    # Select rewards corresponding only to touchdown transitions.
    final_rewards = logs["rewards"][logs["dones"]]

    # Calculate the mean final reward (Monte Carlo landing rate).
    return float(np.mean(final_rewards))

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

