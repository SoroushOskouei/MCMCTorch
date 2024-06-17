# MCMCTorch
A library that focuses on Bayesian Markov Chain Monte Carlo (MCMC) sampling methods using PyTorch. It includes several sampler classes for different MCMC algorithms, utilities for managing sample chains, and support for custom probability distributions.
This library was built on Python==3.12.0, Pytorch==2.3.1, Numpy==1.26.0, Matplotlib==3.9.0.

For citation check: [![DOI](https://zenodo.org/badge/813093740.svg)](https://zenodo.org/doi/10.5281/zenodo.11951534)

## Library Components

### Samplers

#### `BasicSampler`
- **Purpose**: Base class for MCMC samplers.
- **Methods**:
  - `__init__(target_distribution, initial_state)`: Initialize the sampler.
  - `initialize()`: (Re)initialize the Markov chain to the starting state.
  - `step()`: Perform a single MCMC step (to be overridden).
  - `sample(num_samples)`: Generate samples from the target distribution.

#### `MetropolisHastingsSampler(BasicSampler)`
- **Purpose**: Implements the Metropolis-Hastings MCMC algorithm.
- **Methods**:
  - `__init__(target_distribution, initial_state, proposal_distribution)`: Initialize the sampler with a proposal distribution.
  - `proposal(current_state)`: Generate a proposal state based on the current state.
  - `acceptance_probability(current_state, proposed_state)`: Calculate the acceptance probability for the proposed state.
  - `step()`: Perform a single Metropolis-Hastings step.

#### `HamiltonianMonteCarloSampler(BasicSampler)`
- **Purpose**: Implements Hamiltonian Monte Carlo (HMC) sampling.
- **Methods**:
  - `__init__(target_distribution, initial_state, step_size, num_leapfrog_steps)`: Initialize the HMC sampler.
  - `leapfrog(q, p, grad)`: Perform the leapfrog steps for Hamiltonian dynamics.
  - `energy_calculation(q, p)`: Calculate the Hamiltonian energy.
  - `step()`: Perform an HMC step.

#### `NUTSSampler(HamiltonianMonteCarloSampler)`
- **Purpose**: Implements the No-U-Turn Sampler (NUTS), an adaptive form of HMC.
- **Methods**:
  - `__init__(target_distribution, initial_state, max_tree_depth)`: Initialize the NUTS sampler.
  - `find_reasonable_step_size()`: Heuristically find a reasonable initial step size.
  - `step()`: Perform a single NUTS step using an adaptive algorithm.

#### `SliceSampler(BasicSampler)`
- **Purpose**: Implements slice sampling.
- **Methods**:
  - `__init__(target_distribution, initial_state, width, max_steps_out)`: Initialize the slice sampler.
  - `find_interval(y)`: Find an interval that contains a significant amount of the probability mass.
  - `sample_within_interval(L, R, y)`: Sample from the interval where the density is above a level y.
  - `step()`: Perform a slice sampling step.

#### `GibbsSampler`
- **Purpose**: Implements Gibbs sampling for multivariate distributions.
- **Methods**:
  - `__init__(initial_state)`: Initialize the Gibbs sampler.
  - `conditional_sample(index, conditioned_values)`: Sample a variable conditionally.
  - `step()`: Perform one full cycle of Gibbs sampling.

### Utilities

#### `Chain`
- **Purpose**: Manages a chain of samples and acceptance rates.
- **Methods**:
  - `add_sample(sample, accepted)`: Add a sample and update the acceptance rate.
  - `get_samples()`: Retrieve all samples from the chain.
  - `summary_statistics()`: Calculate summary statistics for the samples.

#### `Trace`
- **Purpose**: Manages and visualizes traces of samples.
- **Methods**:
  - `plot_trace()`: Plot the trace of samples.
  - `plot_autocorrelation(max_lag)`: Plot the autocorrelation function.
  - `compute_effective_sample_size()`: Compute the effective sample size.

#### `Model`
- **Purpose**: Manages data, priors, likelihoods, and sampling.
- **Methods**:
  - `add_data(**data)`: Add observational data.
  - `add_prior(param_name, distribution)`: Add a prior distribution for a parameter.
  - `add_likelihood(likelihood_func)`: Set the likelihood function.
  - `run_sampler(sampler_class, num_samples, **sampler_args)`: Run an MCMC sampler to generate posterior samples.

#### `Visualization`
- **Purpose**: Visualization tools for distributions and samples.
- **Methods**:
  -`plot_distribution(distribution, label, num_points, color)`: Plot a distribution.
  -`plot_prior(param_name)`: Plot the prior distribution of a parameter.
  -`plot_posterior(param_name, samples)`: Plot the posterior distribution based on samples.`

### Example Usage

```python

# Example usage:
if __name__ == "__main__":
    # Define a target distribution: Normal(0, 1)
    target_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    
    # Initial state
    initial_state = torch.tensor([0.0], requires_grad=False)
    
    # Create sampler
    sampler = BasicSampler(target_dist, initial_state)
    sampler.initialize()  # Optional: Re-initialize if necessary
    samples = sampler.sample(1000)
    
    print(samples.mean(), samples.std())  # Check if mean and std dev are close to 0 and 1


# Example usage:
if __name__ == "__main__":
    chain = Chain()
    
    # Simulate adding samples (normally you would integrate this with the sampling code)
    for i in range(100):
        sample = torch.randn(1)  # Normally distributed sample
        accepted = torch.rand(1) < 0.75  # 75% acceptance probability
        chain.add_sample(sample, accepted)

    print(chain.get_samples())  # Display all samples
    print(chain.summary_statistics())  # Display summary statistics


# Example usage:
if __name__ == "__main__":
    target_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    initial_state = torch.tensor([0.0])
    proposal_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([0.1]))  # Smaller variance for small proposal jumps

    sampler = MetropolisHastingsSampler(target_dist, initial_state, proposal_dist)
    chain = Chain()

    for _ in range(1000):
        sample, accepted = sampler.step()
        chain.add_sample(sample, accepted)
    
    print(chain.summary_statistics())  # Display summary statistics



# Example usage
if __name__ == "__main__":
    target_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    initial_state = torch.tensor([0.0], requires_grad=True)
    
    sampler = HamiltonianMonteCarloSampler(target_dist, initial_state, step_size=0.1, num_leapfrog_steps=10)
    for _ in range(100):
        sample, accepted = sampler.step()
        print(sample, accepted)


# Example usage:
if __name__ == "__main__":
    target_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ## Ensure that the initial state tensor has requires_grad set to True. 
    ## This is crucial because the Hamiltonian Monte Carlo and No-U-Turn Sampler methods rely heavily on gradients for their computations.
    initial_state = torch.tensor([0.0], requires_grad=True)

    sampler = NUTSSampler(target_dist, initial_state)
    chain = Chain()

    for _ in range(100):
        sample, accepted = sampler.step()
        chain.add_sample(sample, accepted)
    
    print(chain.summary_statistics())  # Display summary statistics


# Example usage:
if __name__ == "__main__":
    target_dist = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    initial_state = torch.tensor([0.0])

    sampler = SliceSampler(target_dist, initial_state)
    samples = [sampler.step() for _ in range(1000)]

    samples_tensor = torch.stack(samples)
    print(samples_tensor.mean(), samples_tensor.std())  # Check if mean and std dev are close to 0 and 1



# Example usage:
if __name__ == "__main__":
    initial_state = torch.tensor([2.0, 3.0])  # Initial state for a 2-variable chain

    sampler = GibbsSampler(initial_state)
    samples = [sampler.step() for _ in range(1000)]

    samples_tensor = torch.stack(samples)
    print(samples_tensor.mean(), samples_tensor.std()) 


# Example usage:
if __name__ == "__main__":
    # Define a simple Gaussian log probability function for demonstration
    def gaussian_log_prob(x):
        mu = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        return -0.5 * ((x - mu) ** 2) / sigma ** 2 - torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi)))

    # Create a custom distribution instance
    custom_dist = CustomDistribution(log_prob_func=gaussian_log_prob)

    # Test the log probability calculation
    x = torch.tensor([0.0], requires_grad=True)
    print("Log Prob at x=0:", custom_dist.log_prob(x))  # Should match the log prob of standard normal at 0
    print("Gradient at x=0:", custom_dist.grad_log_prob(x))  # Should be zero at the mean of the distribution

    # Integrate this custom distribution into an MCMC framework
    # Suppose we are using the Hamiltonian Monte Carlo sampler
    initial_state = torch.tensor([0.5], requires_grad=True)
    sampler = HamiltonianMonteCarloSampler(target_distribution=custom_dist, initial_state=initial_state, step_size=0.1, num_leapfrog_steps=10)
    chain = Chain()

    for _ in range(100):
        sample, accepted = sampler.step()
        chain.add_sample(sample, accepted)
    
    print(chain.summary_statistics())  # Display summary statistics



# Example usage:
if __name__ == "__main__":
    # Generating some example data
    data = torch.randn(1000) + torch.sin(torch.linspace(0, 20, 1000))  # Some correlated data
    trace = Trace(data)

    trace.plot_trace()
    trace.plot_autocorrelation()
    print("Effective Sample Size:", trace.compute_effective_sample_size())



# Example usage:
if __name__ == "__main__":
    model = Model()
    model.add_data(observation=torch.tensor([5.0]))

    # Add prior: Normal(0, 1) for parameter 'mu'
    model.add_prior('mu', torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])))

    # Define a simple likelihood function
    def likelihood(params, data):
        mu = params['mu']
        obs = data[0]['observation']
        return torch.distributions.Normal(mu, torch.tensor([1.0])).log_prob(obs).sum()

    model.add_likelihood(likelihood)

    # Assume MetropolisHastingsSampler has been correctly defined and adapted for this use
    samples = model.run_sampler(MetropolisHastingsSampler, 1000, proposal_distribution=torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.2])))

    # Print or analyze samples
    print(samples)



# Example usage:
if __name__ == "__main__":

    # Set up the model and add prior
    model = Model()
    model.add_prior('mu', torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])))

    # Assume some sampled data for the posterior
    posterior_samples = torch.distributions.Normal(torch.tensor([0.5]), torch.tensor([1.0])).sample((1000,))

    # Initialize visualization
    vis = Visualization(model)

    # Plotting prior and posterior
    vis.plot_prior('mu')
    vis.plot_posterior('mu', posterior_samples)
```
