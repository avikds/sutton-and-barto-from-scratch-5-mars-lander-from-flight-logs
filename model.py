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

# Step 5 - exact_values
class FixedDraw:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def exact_values(kind, tol=1e-12):
    """
    Compute the exact landing probability of every state
    under the specified autopilot using iterative policy evaluation.

    Parameters:
        kind: Autopilot type ('cautious' or 'fast').
        tol: Convergence tolerance.

    Returns:
        A float NumPy array of shape:
        (MAX_ALTITUDE + 1, 2 * MAX_SPEED + 1, MAX_FUEL + 1).

        V[h, v + MAX_SPEED, f] represents the probability
        of landing safely from state (h, v, f).
    """

    # Define the shape of the state-value array.
    value_shape = (
        MAX_ALTITUDE + 1,
        2 * MAX_SPEED + 1,
        MAX_FUEL + 1
    )

    # Enumerate all non-terminal states.
    nonterminal_states = [
        (h, v, f)
        for h in range(1, MAX_ALTITUDE + 1)
        for v in range(-MAX_SPEED, MAX_SPEED + 1)
        for f in range(MAX_FUEL + 1)
    ]

    n_states = len(nonterminal_states)

    # Each state has 3 possible actions and 3 possible gust outcomes.
    n_outcomes = 3 * 3

    # Allocate arrays to store state indices and transition information.
    state_indices = np.empty(n_states, dtype=np.intp)

    next_indices = np.empty(
        (n_states, n_outcomes),
        dtype=np.intp
    )

    rewards = np.empty(
        (n_states, n_outcomes),
        dtype=float
    )

    probabilities = np.empty(
        (n_states, n_outcomes),
        dtype=float
    )

    # Gust values are determined by the FixedDraw input.
    # Their corresponding probabilities are 0.1, 0.8 and 0.1.
    gust_outcomes = (
        (0.05, 0.1),
        (0.50, 0.8),
        (0.95, 0.1)
    )

    # Precompute all possible transitions for every non-terminal state.
    for i, state in enumerate(nonterminal_states):

        h, v, f = state

        # Convert the current state into its flattened array index.
        state_indices[i] = np.ravel_multi_index(
            (h, v + MAX_SPEED, f),
            value_shape
        )

        # Obtain the autopilot's action probability distribution.
        action_probs = autopilot_probs(state, kind)

        outcome_idx = 0

        # Enumerate all three possible thrust actions.
        for action in range(3):

            # Enumerate all three possible gust outcomes.
            for gust_draw, gust_prob in gust_outcomes:

                # Simulate one transition using a fixed gust.
                next_state, reward, done = lander_step(
                    state,
                    action,
                    FixedDraw(gust_draw)
                )

                # Store a terminal index for touchdown.
                # Terminal states always have value zero.
                if done:
                    next_index = 0

                else:
                    # Convert the next non-terminal state
                    # into its flattened array index.
                    next_index = np.ravel_multi_index(
                        (
                            next_state[0],
                            next_state[1] + MAX_SPEED,
                            next_state[2]
                        ),
                        value_shape
                    )

                # Record the next-state index and immediate reward.
                next_indices[i, outcome_idx] = next_index
                rewards[i, outcome_idx] = reward

                # Calculate the joint probability of this outcome:
                # policy action probability * gust probability.
                probabilities[i, outcome_idx] = (
                    action_probs[action] * gust_prob
                )

                outcome_idx += 1

    # Initialize all state values to zero.
    # This also ensures all touchdown states remain terminal.
    values = np.zeros(
        int(np.prod(value_shape)),
        dtype=float
    )

    # Perform synchronous iterative policy evaluation.
    while True:

        # Calculate the expected return for every transition.
        # Values from the previous iteration are used throughout.
        targets = rewards + values[next_indices]

        # Compute the updated value of every non-terminal state.
        updated_nonterminal = np.sum(
            probabilities * targets,
            axis=1
        )

        # Measure the maximum change in state values.
        delta = np.max(
            np.abs(
                updated_nonterminal - values[state_indices]
            )
        )

        # Create a new value array to preserve synchronous updates.
        updated_values = values.copy()

        # Update all non-terminal states simultaneously.
        updated_values[state_indices] = updated_nonterminal

        values = updated_values

        # Stop when the largest value change is below tolerance.
        if delta < tol:
            break

    # Restore the original three-dimensional state-value structure.
    return values.reshape(value_shape)

# Step 6 - polynomial_features
import itertools


def polynomial_features(states, order=1, scale=None):
    """
    Build an order-based polynomial feature representation
    of the lander's telemetry.

    Parameters:
        states: A single state (h, v, f) or an array of states
                with shape (n, 3).
        order: Maximum exponent for each state variable.
        scale: Optional scaling array for the feature columns.

    Returns:
        A float NumPy array of shape (n, (order + 1) ** 3).
    """

    # Convert input states to a floating-point NumPy array.
    states = np.asarray(states, dtype=float)

    # Ensure a single state is represented as a two-dimensional array.
    if states.ndim == 1:
        states = states.reshape(1, -1)

    # Validate that each state contains exactly three components.
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (n, 3) or contain one state.")

    # Generate all exponent combinations in the required order.
    # Each exponent vector is (c_h, c_v, c_f).
    exponents = np.array(
        list(itertools.product(range(order + 1), repeat=3)),
        dtype=int
    )

    # Extract altitude, velocity and fuel for vectorized computation.
    # Shape of states: (n, 3)
    # Shape of exponents: ((order + 1) ** 3, 3)

    # Calculate every polynomial feature:
    # h**c_h * v**c_v * f**c_f.
    features = np.prod(
        states[:, np.newaxis, :] ** exponents[np.newaxis, :, :],
        axis=2
    )

    # Optionally scale each feature column by its corresponding value.
    if scale is not None:
        scale = np.asarray(scale, dtype=float)

        if scale.shape != (features.shape[1],):
            raise ValueError(
                "scale must contain one value for each feature column."
            )

        features = features / scale[np.newaxis, :]

    return features

# Step 7 - fourier_features
def unit_states(states):
    """
    Normalize lander states into the [0, 1] range.

    Parameters:
        states: A single state (h, v, f) or an array
                of states with shape (n, 3).

    Returns:
        A float NumPy array of shape (n, 3).
    """

    # Convert the input into a floating-point NumPy array.
    states = np.asarray(states, dtype=float)

    # Ensure a single state is represented as a 2D array.
    if states.ndim == 1:
        states = states.reshape(1, -1)

    # Validate the state dimensions.
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (n, 3) or contain one state.")

    # Extract altitude, vertical velocity, and fuel.
    h = states[:, 0]
    v = states[:, 1]
    f = states[:, 2]

    # Normalize each state component into the [0, 1] range.
    normalized = np.column_stack([
        h / MAX_ALTITUDE,
        (v + MAX_SPEED) / (2 * MAX_SPEED),
        f / MAX_FUEL
    ])

    return normalized


def fourier_features(states, order):
    """
    Build an order-based Fourier cosine feature representation.

    Parameters:
        states: A single state (h, v, f) or an array
                of states with shape (n, 3).
        order: Maximum integer coefficient for each dimension.

    Returns:
        A float NumPy array of shape (n, (order + 1) ** 3).
    """

    # Normalize the input states before computing Fourier features.
    normalized = unit_states(states)

    # Generate all possible integer coefficient vectors.
    # Each vector is represented as (c_h, c_v, c_f).
    coefficients = np.array(
        list(itertools.product(range(order + 1), repeat=3)),
        dtype=float
    )

    # Calculate the dot product between each normalized state
    # and every coefficient vector.
    dot_products = normalized @ coefficients.T

    # Apply the Fourier cosine transformation.
    # The resulting first feature is always cos(0) = 1.
    features = np.cos(np.pi * dot_products)

    return features

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

