# model from https://juanitorduz.github.io/tfp_lm/, added in UQ part
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tfd = tfp.distributions

"""
y = a + b0x0 + b1x1 + e, e ~ N(0, sigma^2), x0 ~ N(0, 1), x1 ~ N(0, 0.2^2)
#estimate a, b0, b1, sigma

Our model: y ~ N(u, sigma^2), u = a + b0x0 + b1x1
priors: a ~ N(0, 100), b0/b1 ~ N(0, 100), sigma ~ |N(0, 1)|


"""
np.random.seed(42)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]
size = 100
x0 = np.random.randn(size)
x1 = np.random.randn(size) * 0.2
y = alpha + beta[0] * x0 + beta[1] * x1 + np.random.randn(size) * sigma

# Set tensor numeric type.
dtype = 'float32'

x = np.stack([x0, x1], axis=1) # list of [x0, x1]
x = tf.convert_to_tensor(x, dtype=dtype)

y = tf.convert_to_tensor(y, dtype=dtype)
y = tf.reshape(y, (-1, 1)) #change into column vector, each row onw new entry

#define joint distribution
jds_ab = tfd.JointDistributionNamedAutoBatched(dict( 
    sigma=tfd.HalfNormal(scale=[tf.cast(1.0, dtype)]),
    alpha=tfd.Normal(
        loc=[tf.cast(0.0, dtype)], #mean
        scale=[tf.cast(10.0, dtype)] #stdev
    ),
    beta=tfd.Normal(
        loc=[[tf.cast(0.0, dtype)], [tf.cast(0.0, dtype)]], 
        scale=[[tf.cast(10.0, dtype)], [tf.cast(10.0, dtype)]]
    ),
    y=lambda beta, alpha, sigma: 
        tfd.Normal(
            loc=tf.linalg.matmul(x, beta) + alpha, #x.shape = (100, 2), beta.shape = (2, 1), loc.shape = (100, 1)
            scale=sigma
        ) 
))

# Sample from the prior.
prior_samples = jds_ab.sample(500)['y']
prior_samples = tf.squeeze(prior_samples)
prior_mean = tf.math.reduce_mean(prior_samples, axis=0).numpy()
prior_std = tf.math.reduce_std(prior_samples, axis=0).numpy()

# target function
def target_log_prob_fn(beta=beta, alpha=alpha, sigma=sigma):
    return jds_ab.log_prob(beta=beta, alpha=alpha, sigma=sigma, y=y)

# Size of each chain.
num_results = int(1e4)
# Burn-in steps.
num_burnin_steps = int(1e3)
# Hamiltonian Monte Carlo transition kernel. 
# In TFP a TransitionKernel returns a new state given some old state.
hcm_kernel  = tfp.mcmc.HamiltonianMonteCarlo(
  target_log_prob_fn=target_log_prob_fn,
  step_size=1.0,
  num_leapfrog_steps=3
)
# This adapts the inner kernel's step_size.
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
  inner_kernel = hcm_kernel,
  num_adaptation_steps=int(num_burnin_steps * 0.8)
)

# Run the chain (with burn-in). main updating step
@tf.function
def run_chain():
  # Implements MCMC via repeated TransitionKernel steps.
  samples, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[ # starting guesses
          tf.convert_to_tensor([[1.0], [1.0]], dtype=dtype),
          tf.convert_to_tensor([1.0], dtype=dtype), 
          tf.convert_to_tensor([1.0], dtype=dtype)
      ],
      kernel=adaptive_hmc,
      trace_fn=lambda _, pkr: pkr.inner_results.is_accepted #check which ones are accepted/rejected
    )
  return samples

#run sampling
num_chains = 5
chains = [run_chain() for i in range(num_chains)]

# Data processing
chains_t = list(map(list, zip(*chains)))
chains_samples = [tf.squeeze(tf.concat(samples, axis=0)) for samples in chains_t]

chains_df = pd.concat(
    objs=[pd.DataFrame(samples.numpy()) for samples in chains_samples], 
    axis=1
)

params = ['beta_0', 'beta_1', 'alpha', 'sigma']
chains_df.columns = params

chains_df = chains_df \
    .assign(
        sample_id=lambda x: range(x.shape[0]), 
        chain_sample_id=lambda x: x['sample_id'] % num_results,
        chain_id=lambda x: (x['sample_id'] / num_results).astype(int) + 1
    ) \
    .assign(chain_id=lambda x: 'c_' + x['chain_id'].astype(str)) \

chains_df.head()
beta_samples = chains_samples[0] # (50000, 2)
alpha_samples = chains_samples[1] # (50000)
sigma_samples = chains_samples[2] # (50000, )
alpha_samples = tf.reshape(alpha_samples, [-1, 1])
# x.T shape: 2*100
# Epistemic: Var(lambda | data)

lambda_samples = tf.linalg.matmul(beta_samples, tf.transpose(x)) + alpha_samples
"""
Posterior sample 0: beta0_0, beta1_0, alpha_0 → predictions for all 100 x points
Posterior sample 1: beta0_1, beta1_1, alpha_1 → predictions for all 100 x points
...
Posterior sample 49999 → predictions for all 100 x points
So lambda_samples[i, j] = predicted mean at x[j] using posterior sample i.
(50000, 100)
aleatoric uncertainty: average of each row's predicted sigma values
epistemic uncertainty: variance over all row's averages/means
"""
epistemic_var_per_point = tf.math.reduce_variance(lambda_samples, axis=0) #gives 1 variance (over 50000 posteriors) * 100 points
epistemic_var = np.mean(epistemic_var_per_point) # variance of sigma
print(f"Epistemic uncertainty is {tf.sqrt(epistemic_var)}")
# i guess the uncertainty assosicate with the error between true sigma and estimated sigma is accounted for in epistemic uncertainty

# Aleatoric: E[sigma^2 | data]
aleatoric_var = np.mean(sigma_samples**2) # mean of sigma
print(f"Aleatoric uncertainty is {tf.sqrt(aleatoric_var)}")
# Total predictive variance
total_var = epistemic_var + aleatoric_var
print(f"total variance {total_var}")
print(f"total stdev / uncertainty {tf.sqrt(total_var)}")
# print(epistemic_var_per_point), epistemic uncertainty varies with x

"""
result
Epistemic uncertainty, averaged over all x values is 0.18510089814662933
Aleatoric uncertainty is 1.0855611562728882
Total variance 1.2148849964141846
total stdev / uncertainty 1.1022182703018188
"""

#generate samples from posterior
# refer to link