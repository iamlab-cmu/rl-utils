from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

def reps_solve_info_analysis(input_path):
    
    path_to_reps_info = Path(input_path)
    assert path_to_reps_info.exists(), (
        f"Expected input_path \"{path_to_reps_info}\" to exist, but it does not."
    )

    with open(path_to_reps_info, "rb") as f:
        reps_solve_info = pickle.load(f)
    
    # TODO: change to dict-like
    assert isinstance(reps_solve_info, dict), (
        f"Expected reps_solve_info to be a dict, but it is a {type(reps_solve_info)}."
    )

    mean_param_hist = reps_solve_info['history']['policy_params_mean']
    var_diag_param_hist = reps_solve_info['history']['policy_params_var_diag']
    mean_rew_hist = reps_solve_info['history']['mean_reward']
    import pdb; pdb.set_trace()
    