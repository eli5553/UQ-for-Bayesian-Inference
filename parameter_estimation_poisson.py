# model from https://juanitorduz.github.io/tfp_hcm/?utm_source=chatgpt.com,  added in UQ part
# parameter estimation for possion distribution
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

tfd = tfp.distributions

def plot_discrete(samples, x_label, y_label):
    y_range, _, c = tf.unique_with_counts(samples)
    y_range = y_range.numpy()
    c = c.numpy()

    plt.bar(y_range, c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # integers on x-axis
    plt.show()

def plot_continuous(samples, x_label, y_label):
    plt.hist(samples.numpy(), bins=50, density=True, alpha=0.6, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


#true samples
tf.random.set_seed(seed=42)
# Number of samples. 
n = 100
# True rate parameter. 
rate_true = 2.0
# Define Poisson distribution with the true rate parameter. 
poisson_true = tfd.Poisson(rate=rate_true)
# Generate samples.
poisson_samples = poisson_true.sample(sample_shape=n)
# poisson_samples 
# plot_discrete(poisson_samples, 'Poisson outcomes', 'Counts')


# Define parameters for the prior distribution. 
a = 4.5
b = 2
# Define prior distribution. 
gamma_prior = tfd.Gamma(concentration=a, rate=b)

# Generate samples. 
gamma_prior_samples = gamma_prior.sample(sample_shape=1e4)
# print(gamma_prior_samples)
# plot_continuous(gamma_prior_samples, 'Gamma outcomes', 'Density')

sample_mean = tf.reduce_mean(gamma_prior_samples)
sample_median = tfp.stats.percentile(x=gamma_prior_samples, q=50)
# print(sample_mean, sample_median)


def build_model(a=4.5, b=2):
    # Prior Distribution.
    rate = tfd.Gamma(concentration=a, rate=b) # a distribution
    # Likelihood: Independent samples of a Poisson distribution. 
    observations = lambda rate: tfd.Sample(
        distribution=tfd.Poisson(rate=rate), #TFP will draw samples from rate and then plug those samples into the Poisson distribution to compute likelihoods or log-probabilitie
        sample_shape=len(poisson_samples) #sample_shape (N) independent samples, each with possion param (lambda) rate
    )
    return tfd.JointDistributionNamed(dict(rate=rate, obs=observations))

def target_log_prob_fn(rate):
    model = build_model()
    return model.log_prob(rate=rate, obs=poisson_samples)

# Define rates range.
rates = np.linspace(start=0.01, stop=10.0, num=1000)

# Compute joint-log-probability.
model_log_probs = np.array([
    target_log_prob_fn(rate).numpy() 
    for rate in rates
])

# Get rate which maximizes the log-probability of the model. simple, pointwise
#simple numerical approximation, finds the best parameter (rate) according to log likelihood function
log_prob_maximizer = rates[np.argmax(model_log_probs)]

"""MC (and other MCMC methods) samples from the full posterior:
p(parameters|data)
From the samples, you can compute:
Posterior mean and median.
Posterior standard deviation → epistemic uncertainty about the parameters.

Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm that takes a series of gradient-informed steps to produce a Metropolis proposal.
details: https://arxiv.org/abs/1701.02434"""

# Size of each chain, number of posterior samples
num_results = int(1e4)
# Burn-in steps, Number of initial steps to discard
num_burnin_steps = int(1e3)
# Hamiltonian Monte Carlo transition kernel. 
# In TFP a TransitionKernel returns a new state given some old state.
hcm_kernel  = tfp.mcmc.HamiltonianMonteCarlo(
  target_log_prob_fn=target_log_prob_fn,
  step_size=1.0,
  num_leapfrog_steps=3 # Each call proposes a new point in parameter space. Accepts or rejects the proposal using the Metropolis criterion.
  
)
# This adapts the inner kernel's step_size.
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
  inner_kernel = hcm_kernel,
  num_adaptation_steps=int(num_burnin_steps * 0.8)
)
# Run the chain (with burn-in).
@tf.function
def run_chain():
  # Run the chain (with burn-in). 
  # Implements MCMC via repeated TransitionKernel steps.
  samples, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=1.0,
      kernel=adaptive_hmc, #transition kernel defining how new states are proposed
      trace_fn=lambda _, pkr: pkr.inner_results.is_accepted #extra info to return: returns a boolean array saying whether each proposed state was accepted.
    )
  return samples

# Set number of chains. 
num_chains = 5
# chains is a Python list of length 5, each element is a (10000,) tensor. so 5 rows*10000 columns
chains = [run_chain() for i in range(num_chains)]

# We store the samples in a pandas dataframe.
chains_df = pd.DataFrame([t.numpy() for t in chains])
# print("chains df")
# print(chains_df)
chains_df = chains_df.T.melt(var_name='chain_id', value_name='sample') # T is transpose
# after T: 10000 rows * 5 columns. after melt: more rows, just two columns
# chains_df.head() #chain id, sample as columns
chain_samples_mean = []
chain_samples_std = []
for i in range(5):
    chain_samples = chains_df.query(f'chain_id == {i}').reset_index(drop=True)['sample']
    chain_samples_mean.append(chain_samples.mean())
    chain_samples_std.append(chain_samples.std())

print(chain_samples_mean)
print(chain_samples_std)

"""
Var(y⋆)=aleatoric (E[λ|data]) + epistemic (Var(λ|data))
"""
all_samples = np.concatenate([chain.numpy() for chain in chains])   # shape (num_chains * num_results,)

post_mean = np.mean(all_samples)         # E[λ | data], aleatoric var
post_var  = np.var(all_samples, ddof=1)  # Var(λ | data), epistemic var
total_var = post_mean + post_var # usually: E[Var(y|theta, D)] + Var(E[y|theta, D])
total_std = np.sqrt(total_var)
print("Posterior mean / Aleatoric variance (E[λ|data]):", post_mean)
print("Posterior var / Epistemic variance:", post_var)
print("Posterior 90% CI:", np.percentile(all_samples, [5, 95]))
print("Total predictive var (sqrt(E[λ] + Var(λ))):", total_var)
print("Total predictive std (sqrt(E[λ] + Var(λ))):", total_std)

# Empirical posterior predictive intervals by simulation
nsim = 10000
# sample lambda from posterior (nsim draws)
lambda_samps = np.random.choice(all_samples, size=nsim, replace=True)
# draw Poisson outcomes conditional on each sampled lambda
y_pred_samps = np.random.poisson(lam=lambda_samps)
# get predictive mean and 90% interval for y*
print("Predictive mean (E[y*]):", y_pred_samps.mean())
print("Predictive 90% interval (y*):", np.percentile(y_pred_samps, [5,95]))


# sample from posterior
y_post_pred = tfd.Poisson(rate=chains_df['sample']).sample(1)
y_post_pred  = tf.reshape(y_post_pred, [-1])

plot_discrete(y_post_pred, "values", "number of samples")