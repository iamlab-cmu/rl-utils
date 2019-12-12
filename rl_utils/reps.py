import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------

# input either list/array of dimension []
# always returns np type of size 
def reps_weights_from_rewards(rewards, epsilon, min_eta):
    """
    REPS optimization function to calculate weights based on rewards.

    Args:
        rewards (list of float or np.ndarray): An array of rewards
            of length n_rewards
        epsilon (float): Kullback-Leibler (KL) divergence bound
        min_eta (float): Minimum value of the `temperature'
            parameter that the optimization function may return

    Returns:
        weights (np.ndarray, shape=(n_rewards,)): Weights of the
            rewards based on relative entropy optimization
        eta (float): Value of the 'temperature' parameter
            as output from the optimization function
    """

    R = rewards - np.max(rewards)

    def g(eta):
        return eta*np.log(np.mean(np.exp(R/eta))) + np.max(rewards) + eta*epsilon

    def gp(eta):
        return np.log(np.mean(np.exp(R/eta))) - (np.mean(np.exp(R/eta) * (R/eta)) / np.mean(np.exp(R/eta))) + epsilon

    eta_0 = 0.1
    while True:
        res = minimize(g, eta_0, jac=gp, constraints=({'type': 'ineq', 'fun': lambda eta: eta - min_eta}), method='SLSQP')
        eta = res.x[0]

        if np.isnan(eta):
            eta_0 = np.random.rand() + min_eta
            continue
        break

    weights = np.exp(R/eta)

    return weights, eta


class Reps(object):
    """
    Relative Entropy Policy Search (REPS) algorithm class.

    Args:
        epsilon (float): Kullback-Leibler (KL) divergence bound
        min_eta (float): Minimum value of the `temperature'
            parameter that the optimization function may return
        policy_variance_model (str): Modeling assumption for the
            variance of the policy parameters. Two options:
                - 'standard' returns full covariance matrix
                - 'diagonal' assumes the policy parameters are
                    completely uncorrelated
    """

    def __init__(self, epsilon, min_eta, policy_variance_model='standard'):
        self.epsilon = epsilon
        self.min_eta = min_eta
        self.policy_variance_model = policy_variance_model

    def weights_from_rewards(self, rewards):
        """ Wrapper function for reps_weights_from_rewards """
        return reps_weights_from_rewards(rewards, self.epsilon, self.min_eta)

    def policy_from_samples_and_rewards(self, policy_param_samples, rewards):
        """
        Calculates a new policy using REPS based on policy parameter samples
        and rewards obtained when using these samples.

        Args:
            policy_param_samples (np.ndarray, shape=(n_samples, n_params)):
                An array of samples from a parameterized policy
            rewards (list of float or np.ndarray): An array of rewards
                of length n_samples

        Returns:
            policy_params_mean (np.ndarray, shape=(n_params,)):
                An array of the mean for each policy parameter,
                or None if policy calculation failed.
            policy_params_var (np.ndarray):
                An ndarray of the covariance for the policy parameters,
                or None if policy calculation failed.
                Shape depends on the value of policy_variance_model
            eta (float): Value of the 'temperature' parameter
                as output from the REPS optimization function
        """

        # Input argument handling
        n_samples = len(policy_param_samples)
        n_params = len(policy_param_samples[0])

        assert n_samples == len(rewards), \
            "Expected the length of policy_param_samples and rewards to be the same, but they are not."

        # Calculate weights
        weights, eta = self.weights_from_rewards(rewards)

        # Catch if optimization returned an inf
        if np.any(weights == np.inf):
            policy_params_mean = None
            policy_params_var = None
        else:
            # Update policy mean
            new_mean = np.zeros(n_params)
            for param_sample, weight in zip(policy_param_samples, weights):
                new_mean += param_sample*weight
            sum_weights = np.sum(weights)
            policy_params_mean = new_mean/sum_weights

            # Update policy variance
            if self.policy_variance_model == 'standard':
                # Full covariance matrix
                var_shape = (n_params, n_params)
            elif self.policy_variance_model == 'diagonal':
                # One dimensional uncorrelated variance
                var_shape = n_params
            new_var = np.zeros(var_shape)

            for param_sample, weight in zip(policy_param_samples, weights):
                policy_params_mean_diff = param_sample - policy_params_mean
                if self.policy_variance_model == 'standard':
                    new_var += np.outer(policy_params_mean_diff,policy_params_mean_diff)*weight
                elif self.policy_variance_model == 'diagonal':
                    new_var += np.power(policy_params_mean_diff,2.)*weight
            policy_params_var = new_var/sum_weights

        return policy_params_mean, policy_params_var, eta
