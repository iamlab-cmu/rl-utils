import pytest

import rl_utils

# ---------------------------------------------------------

def test_reps():

    # Create REPS object
    epsilon = 0.5
    min_eta = 0.0001

    reps = rl_utils.Reps(epsilon=epsilon,
                         min_eta=min_eta)
    print('in test_reps')

    rewards = [x for x in range(10)]
    print(rewards)

    weights = reps.get_weights(rewards)
    print(weights)
    import pdb
    pdb.set_trace()