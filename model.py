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

# Step 8 - rbf_features
def rbf_features(states, centers, sigma):
    """
    Compute Gaussian radial-basis features for normalized lander states.

    Parameters:
        states: A single state (h, v, f) or an array of states
                with shape (n, 3).
        centers: Array of RBF centers with shape (k, 3),
                 defined in the normalized state space.
        sigma: Positive Gaussian width parameter.

    Returns:
        A float NumPy array of shape (n, k).
    """

    # Normalize the input states into the [0, 1] range.
    normalized = unit_states(states)

    # Convert the RBF centers into a floating-point NumPy array.
    centers = np.asarray(centers, dtype=float)

    # Validate the dimensions of the centers.
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (k, 3).")

    # Ensure the Gaussian width is strictly positive.
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    # Calculate the difference between each normalized state
    # and every RBF center using broadcasting.
    differences = (
        normalized[:, np.newaxis, :]
        - centers[np.newaxis, :, :]
    )

    # Compute squared Euclidean distances for every state-center pair.
    squared_distances = np.sum(
        differences ** 2,
        axis=2
    )

    # Apply the Gaussian radial-basis function.
    features = np.exp(
        -squared_distances / (2.0 * sigma ** 2)
    )

    return features

# Step 9 - tile_features
def tile_features(states, n_tilings, tiles_per_dim):
    """
    Build a tile-coded feature representation of normalized lander states.

    Parameters:
        states: A single state (h, v, f) or an array of states
                with shape (n, 3).
        n_tilings: Number of offset grids.
        tiles_per_dim: Number of tiles along each state dimension.

    Returns:
        A binary float NumPy array of shape
        (n, n_tilings * (tiles_per_dim + 1) ** 3).

        Exactly one tile is active per tiling for every state.
    """

    # Normalize the lander states into the [0, 1] range.
    normalized = unit_states(states)

    # Validate the tiling parameters.
    if n_tilings <= 0:
        raise ValueError("n_tilings must be positive.")

    if tiles_per_dim <= 0:
        raise ValueError("tiles_per_dim must be positive.")

    # Number of tiles along each dimension.
    k = tiles_per_dim

    # Each tiling contains (k + 1)^3 features.
    # The extra tile in each dimension absorbs grid offsets.
    tiles_per_tiling = (k + 1) ** 3

    # Allocate the complete binary feature matrix.
    n_samples = normalized.shape[0]

    features = np.zeros(
        (n_samples, n_tilings * tiles_per_tiling),
        dtype=float
    )

    # Asymmetric displacement vector specified by Sutton and Barto.
    displacement = np.array([1, 3, 5])

    # Row indices used to activate one feature per state.
    rows = np.arange(n_samples)

    # Construct each offset tiling.
    for t in range(n_tilings):

        # Calculate the offset for the current tiling.
        offset = (
            (t * displacement) % n_tilings
        ) / (n_tilings * k)

        # Calculate the tile index along each state dimension.
        # Indices are clipped to k to account for the extra boundary tile.
        tile_indices = np.minimum(
            np.floor((normalized + offset) * k),
            k
        ).astype(int)

        # Extract the individual dimension indices.
        i_h = tile_indices[:, 0]
        i_v = tile_indices[:, 1]
        i_f = tile_indices[:, 2]

        # Flatten the three-dimensional tile coordinates.
        local_indices = (
            (i_h * (k + 1) + i_v) * (k + 1) + i_f
        )

        # Assign each tiling its own consecutive feature block.
        active_indices = (
            t * tiles_per_tiling + local_indices
        )

        # Activate exactly one tile for every state in this tiling.
        features[rows, active_indices] = 1.0

    return features

# Step 10 - step_size_rule
def step_size_rule(X, tau):
    """
    Calculate the recommended step size for a linear function
    approximation method using Sutton and Barto's rule of thumb.

    Parameters:
        X: Feature matrix of shape (n, d), where each row
           represents one feature vector.
        tau: Expected number of presentations of a feature vector.

    Returns:
        Python float representing the recommended step size.
    """

    # Convert the feature matrix into a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Calculate the squared Euclidean norm of each feature vector.
    squared_norms = np.sum(X ** 2, axis=1)

    # Estimate the expected squared norm using the mean across all rows.
    mean_squared_norm = np.mean(squared_norms)

    # Apply Sutton and Barto's recommended step-size rule.
    alpha = 1.0 / (tau * mean_squared_norm)

    return float(alpha)

# Step 11 - semi_gradient_td
def semi_gradient_td(X, X_next, rewards, dones, alpha, rho=None, gamma=1.0):
    """
    Run one online pass of linear semi-gradient TD(0).

    Parameters:
        X: Feature matrix of current states, shape (T, d).
        X_next: Feature matrix of next states, shape (T, d).
        rewards: Reward array, shape (T,).
        dones: Boolean array indicating terminal transitions, shape (T,).
        alpha: Step size (learning rate).
        rho: Optional per-transition importance sampling ratios.
             Defaults to 1 for every transition.
        gamma: Discount factor, defaulting to 1.0.

    Returns:
        Final weight vector w of shape (d,).
    """

    # Convert inputs to NumPy arrays without modifying the originals.
    X = np.asarray(X, dtype=float)
    X_next = np.array(X_next, dtype=float, copy=True)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)

    # Extract the number of transitions and feature dimensions.
    T, d = X.shape

    # Validate the input dimensions.
    if X_next.shape != (T, d):
        raise ValueError("X_next must have the same shape as X.")

    if rewards.shape != (T,):
        raise ValueError("rewards must have shape (T,).")

    if dones.shape != (T,):
        raise ValueError("dones must have shape (T,).")

    # Use unit importance sampling ratios for on-policy learning.
    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)

        if rho.shape != (T,):
            raise ValueError("rho must have shape (T,).")

    # Terminal states have zero value, so zero their next-state features.
    X_next[dones] = 0.0

    # Initialize the weight vector with zeros.
    w = np.zeros(d, dtype=float)

    # Process transitions sequentially using online updates.
    for t in range(T):

        # Calculate the TD error using the current weight vector.
        delta = (
            rewards[t]
            + gamma * np.dot(w, X_next[t])
            - np.dot(w, X[t])
        )

        # Apply the semi-gradient TD(0) weight update.
        w += alpha * rho[t] * delta * X[t]

    return w

# Step 12 - lstd
def lstd(X, X_next, rewards, dones, epsilon=1e-3, rho=None, gamma=1.0):
    """
    Compute the TD fixed-point solution using Least-Squares TD (LSTD).

    The inverse matrix is maintained incrementally using the
    Sherman-Morrison formula.

    Parameters:
        X: Feature matrix of current states, shape (T, d).
        X_next: Feature matrix of next states, shape (T, d).
        rewards: Reward array, shape (T,).
        dones: Boolean array indicating terminal transitions, shape (T,).
        epsilon: Positive regularization parameter.
        rho: Optional per-transition importance sampling ratios.
             Defaults to 1 for every transition.
        gamma: Discount factor, defaulting to 1.0.

    Returns:
        Weight vector of shape (d,).
    """

    # Convert inputs to floating-point arrays without modifying originals.
    X = np.asarray(X, dtype=float)
    X_next = np.array(X_next, dtype=float, copy=True)
    rewards = np.asarray(rewards, dtype=float)
    dones = np.asarray(dones, dtype=bool)

    # Extract the number of transitions and feature dimensions.
    T, d = X.shape

    # Validate input dimensions.
    if X_next.shape != (T, d):
        raise ValueError("X_next must have the same shape as X.")

    if rewards.shape != (T,):
        raise ValueError("rewards must have shape (T,).")

    if dones.shape != (T,):
        raise ValueError("dones must have shape (T,).")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    # Default to unit importance sampling ratios.
    if rho is None:
        rho = np.ones(T, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float)

        if rho.shape != (T,):
            raise ValueError("rho must have shape (T,).")

    # Terminal next states have zero value.
    X_next[dones] = 0.0

    # Initialize the inverse regularized matrix.
    A_inv = np.eye(d, dtype=float) / epsilon

    # Initialize the reward-weighted feature vector.
    b = np.zeros(d, dtype=float)

    # Process every transition in chronological order.
    for t in range(T):

        # Current-state feature vector.
        x = X[t]

        # Define the second vector of the rank-one matrix update.
        # The corresponding matrix contribution is x @ u.T.
        u = rho[t] * (x - gamma * X_next[t])

        # Compute A_inv @ x.
        A_inv_x = A_inv @ x

        # Compute A_inv.T @ u for the Sherman-Morrison update.
        # This preserves the correct nonsymmetric LSTD matrix.
        v = A_inv.T @ u

        # Sherman-Morrison rank-one inverse update.
        denominator = 1.0 + v @ x

        A_inv -= np.outer(A_inv_x, v) / denominator

        # Accumulate the reward-weighted feature vector.
        b += rho[t] * rewards[t] * x

    # Return the LSTD solution.
    return A_inv @ b

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

