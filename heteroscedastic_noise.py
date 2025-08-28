# parameter estimation for y = xsin(x) +xe1 +e2
# compare this and the NN and BNN
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from generate_data import get_data

tfd = tfp.distributions

"""
y = xsin(x) +xe1 +e2
e1 ~ N(0, 0.6^2), e2 ~ N(0, 0.8^2)
estimate 0.6 and 0.8 (sigma1 and sigma2)

model: y ~ N(μθ​(x),sigma^2(x)), μθ​(x) = xsinx, sigma^2(x) = sigma1^2 * x^2 + sigma2^2
priors:
sigma1 ~ HalfNormal(1.0) / |N(0, 1)|    # multiplicative noise
sigma2 ~ HalfNormal(1.0)    # additive noise

"""
np.random.seed(42)

# true parameter values
sigma1 = 0.6
sigma2 = 0.8
size = 1000
x, y = get_data(sigma1, sigma2, n = size)

# true x, y values
dtype = 'float32'
x = tf.convert_to_tensor(x, dtype=dtype) # (1000, 1)
y = tf.convert_to_tensor(y, dtype=dtype) # (1000, 1)
y = tf.reshape(y, (-1, 1)) # just in case

#define joint distribution
jds_ab = tfd.JointDistributionNamedAutoBatched(dict( 
    sigma1=tfd.HalfNormal(scale=[tf.cast(1.0, dtype)]),
    sigma2=tfd.HalfNormal(scale=[tf.cast(1.0, dtype)]),
    y=lambda sigma1, sigma2: 
        tfd.Normal(
            loc=x*tf.sin(x),
            scale=tf.sqrt(sigma1**2 * x**2 + sigma2**2)
        )
))

# prior samples, not used
prior_samples = jds_ab.sample(500)['y']
prior_samples = tf.squeeze(prior_samples)
prior_mean = tf.math.reduce_mean(prior_samples, axis=0).numpy()
prior_std = tf.math.reduce_std(prior_samples, axis=0).numpy()

# target function
def target_log_prob_fn(sigma1, sigma2):
    return jds_ab.log_prob(sigma1=sigma1, sigma2=sigma2, y=y)

# initialise size
num_results = int(1e4)
num_burnin_steps = int(1e3)
# HMC
hcm_kernel  = tfp.mcmc.HamiltonianMonteCarlo(
  target_log_prob_fn=target_log_prob_fn,
  step_size=1.0, # can change
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
          tf.convert_to_tensor([1.0], dtype=dtype), #sigma1 and sigma2
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

sigma1_samples = chains_samples[0] # (50000, )
sigma2_samples = chains_samples[1] # (50000, )
x_sorted = tf.reshape(tf.sort(x, axis=0) , (1, -1)) # (1, 1000)
x_flat = tf.reshape(x_sorted, [-1])  # shape (1000,)
sigma1_reshaped = tf.reshape(sigma1_samples, (-1, 1)) 
sigma2_reshaped = tf.reshape(sigma2_samples, (-1, 1)) # shape (50000, 1)
print(f"estimated sigma1 {tf.reduce_mean(sigma1_samples, axis=0)}")
print(f"estimated sigma2 {tf.reduce_mean(sigma2_samples, axis=0)}")

# Epistemic: Var(lambda | data)
final_y =  x_sorted * np.sin(x_sorted) + sigma1_reshaped * x_sorted + sigma2_reshaped # (50000, 1000)
epistemic_var_row = tf.math.reduce_variance(final_y, axis=0) # var over 50000 param
epistemic_stdev_row = np.sqrt(epistemic_var_row) 
epistemic_var = np.mean(epistemic_var_row)
epistemic_stdev = np.sqrt(epistemic_var)
print(f"Average epistemic uncertainty is {epistemic_stdev}")

# # Aleatoric: E[sigma^2 | data]
final_var = sigma1_reshaped**2 * x_sorted**2 + sigma2**2 # （50000，1000) each row is same param, diff x value
a_var_row = tf.reduce_mean(final_var, axis=0)
a_stdev_row = np.sqrt(a_var_row)

total_var_row = epistemic_var_row + a_var_row
total_var = np.mean(total_var_row)
total_stdev_row = np.sqrt(total_var_row)
total_stdev = np.sqrt(total_var)

# Plot aleatoric variance
plt.figure(figsize=(8,5))
plt.plot(x_flat.numpy(), a_stdev_row, color='red', label='Aleatoric uncertainty (stdev)')
plt.plot(x_flat.numpy(), epistemic_stdev_row, color='red', label='epistemic uncertainty (stdev)')
#plt.plot(x_flat.numpy(), total_stdev_row, color='green', label='total uncertainty (stdev)')
# variances
# plt.plot(x_flat.numpy(), a_var_row.numpy(), color='blue', label='Aleatoric variance')
# plt.plot(x_flat.numpy(), epistemic_var_row.numpy(), color='blue', label='epistemic variance')
# plt.plot(x_flat.numpy(), total_var_row.numpy(), color='green', label='total variance')
plt.xlabel('x')
plt.ylabel('Variance')
plt.title('Aleatoric variance vs x')
plt.legend()
plt.show()


"""
result
estimated sigma1 0.5984596610069275
estimated sigma2 0.7721677422523499
Average epistemic uncertainty is 0.09481338411569595
and graph
- aleatoric uncertainty increases with x
- epistemic uncertainty does not change wrt x, and is small compared to aleatoric uncertainty
"""

