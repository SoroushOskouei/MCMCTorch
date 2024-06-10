import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt

class BasicSampler:
    def __init__(self, target_distribution, initial_state):
        """
        Initialize the sampler with a target distribution and an initial state.
        
        :param target_distribution: A PyTorch distribution object representing the target distribution.
        :param initial_state: A tensor representing the initial state of the Markov chain.
        """
        self.target_distribution = target_distribution
        self.current_state = initial_state

    def initialize(self):
        """
        Initialize or re-initialize the Markov chain to the starting state.
        This can be overridden to perform any setup required before starting the sampling process.
        """
        self.current_state = torch.randn_like(self.current_state)  # Random initialization

    def step(self):
        """
        Perform a single MCMC step using some MCMC algorithm, such as Metropolis-Hastings.

        This method should be overridden by subclasses that specify the exact MCMC method.
        """
        proposed_state = self.current_state + torch.randn_like(self.current_state) * 0.1
        acceptance_probability = min(1, torch.exp(self.target_distribution.log_prob(proposed_state) - 
                                                  self.target_distribution.log_prob(self.current_state)))

        if torch.rand(1) < acceptance_probability:
            self.current_state = proposed_state
        
    def sample(self, num_samples):
        """
        Generate samples from the target distribution.
        
        :param num_samples: The number of samples to generate.
        :return: A tensor containing the generated samples.
        """
        samples = []
        for _ in range(num_samples):
            self.step()
            samples.append(self.current_state.clone())

        return torch.stack(samples)


class Chain:
    def __init__(self):
        """
        Initialize the Chain class with empty lists to store samples and acceptance rates.
        """
        self.samples = []
        self.acceptance_rates = []
    
    def add_sample(self, sample, accepted):
        """
        Add a sample to the chain and update the acceptance rate.
        
        :param sample: A tensor representing the new sample to be added.
        :param accepted: A boolean indicating whether the sample was accepted.
        """
        self.samples.append(sample.clone())
        # Track acceptance as 1 or 0, which will help in calculating the acceptance rate easily.
        self.acceptance_rates.append(float(accepted))
    
    def get_samples(self):
        """
        Retrieve all the samples from the chain.
        
        :return: A tensor of all samples in the chain.
        """
        return torch.stack(self.samples)
    
    def summary_statistics(self):
        """
        Calculate and return summary statistics for the samples in the chain.
        
        :return: A dictionary containing the mean, standard deviation, and acceptance rate of the samples.
        """
        samples_tensor = torch.stack(self.samples)
        mean = samples_tensor.mean().item()
        std_dev = samples_tensor.std().item()
        acceptance_rate = sum(self.acceptance_rates) / len(self.acceptance_rates)
        return {
            "mean": mean,
            "standard deviation": std_dev,
            "acceptance rate": acceptance_rate
        }


class MetropolisHastingsSampler(BasicSampler):
    def __init__(self, target_distribution, initial_state, proposal_distribution):
        """
        Initialize the Metropolis-Hastings sampler with a target distribution, initial state, and proposal distribution.
        
        :param target_distribution: A PyTorch distribution object representing the target distribution.
        :param initial_state: A tensor representing the initial state of the Markov chain.
        :param proposal_distribution: A PyTorch distribution used for proposing new states.
        """
        super().__init__(target_distribution, initial_state)
        self.proposal_distribution = proposal_distribution

    def proposal(self, current_state):
        """
        Generate a proposal state based on the current state.
        
        :param current_state: The current state of the Markov chain.
        :return: A proposed state.
        """
        # Generate a new proposed state by sampling from the proposal distribution centered at the current state.
        proposed_state = self.proposal_distribution.mean + current_state + self.proposal_distribution.sample()
        return proposed_state

    def acceptance_probability(self, current_state, proposed_state):
        """
        Calculate the acceptance probability for moving from the current state to the proposed state.
        
        :param current_state: The current state tensor.
        :param proposed_state: The proposed state tensor.
        :return: The acceptance probability.
        """
        # Calculate the probability of the proposed state and the current state.
        proposed_prob = self.target_distribution.log_prob(proposed_state)
        current_prob = self.target_distribution.log_prob(current_state)

        # Return the acceptance probability.
        return torch.exp(proposed_prob - current_prob)

    def step(self):
        """
        Perform a single Metropolis-Hastings step.
        """
        proposed_state = self.proposal(self.current_state)
        accept_prob = self.acceptance_probability(self.current_state, proposed_state)

        if torch.rand(1) < accept_prob:
            self.current_state = proposed_state
            accepted = True
        else:
            accepted = False
        
        return self.current_state, accepted


class HamiltonianMonteCarloSampler(BasicSampler):

    def __init__(self, target_distribution, initial_state, step_size, num_leapfrog_steps):
        """
        Initialize the Hamiltonian Monte Carlo sampler with a target distribution, initial state, step size, and number of leapfrog steps.
        
        :param target_distribution: A PyTorch distribution object representing the target distribution.
        :param initial_state: A tensor representing the initial state of the Markov chain.
        :param step_size: A float representing the step size in the leapfrog integration.
        :param num_leapfrog_steps: An integer representing the number of leapfrog steps to perform for each sample.
        """
        super().__init__(target_distribution, initial_state)
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps

    def leapfrog(self, q, p, grad):
        """
        Perform the leapfrog steps for Hamiltonian dynamics.
        
        :param q: The current position tensor (state).
        :param p: The current momentum tensor.
        :param grad: The gradient of the Hamiltonian w.r.t. the position at the start.
        :return: The new position and momentum tensors.
        """
        q = q.clone().detach().requires_grad_(True)  # Ensure q tracks gradients
        for _ in range(self.num_leapfrog_steps - 1):
            p -= 0.5 * self.step_size * grad
            q = q + self.step_size * p
            grad = torch.autograd.grad(self.energy_calculation(q, p), q, create_graph=True)[0]
        q = q + self.step_size * p
        p -= 0.5 * self.step_size * grad
        return q, p

    def energy_calculation(self, q, p):
        kinetic_energy = 0.5 * p.pow(2).sum()
        potential_energy = -self.target_distribution.log_prob(q).sum()
        return kinetic_energy + potential_energy

    def step(self):
        p_current = torch.randn_like(self.current_state)
        q_current = self.current_state
        current_grad = torch.autograd.grad(self.energy_calculation(q_current, p_current), q_current, create_graph=True)[0]
        q_proposed, p_proposed = self.leapfrog(q_current, p_current, current_grad)
        p_proposed = -p_proposed
        current_energy = self.energy_calculation(q_current, p_current)
        proposed_energy = self.energy_calculation(q_proposed, p_proposed)
        accept_prob = torch.exp(current_energy - proposed_energy)
        if torch.rand(1) < accept_prob:
            self.current_state = q_proposed
            accepted = True
        else:
            accepted = False
        return self.current_state, accepted


class NUTSSampler(HamiltonianMonteCarloSampler):
    def __init__(self, target_distribution, initial_state, max_tree_depth=10):
        """
        Initialize the No-U-Turn Sampler with a target distribution, initial state, and maximum tree depth.
        
        :param target_distribution: A PyTorch distribution object representing the target distribution.
        :param initial_state: A tensor representing the initial state of the Markov chain.
        :param max_tree_depth: Maximum depth of the binary tree for trajectory expansion.
        """
        super().__init__(target_distribution, initial_state, step_size=1.0, num_leapfrog_steps=1)  # Start with defaults
        self.max_tree_depth = max_tree_depth
        self.step_size = 0.1  # Initial step size, needs tuning
        self.tuning = True

    def find_reasonable_step_size(self):
        """
        Heuristically find a reasonable initial step size.
        """
        p = torch.randn_like(self.current_state)
        current_energy = self.energy_calculation(self.current_state, p)
        proposed_state, proposed_momentum = self.leapfrog(self.current_state, p, torch.autograd.grad(current_energy, self.current_state)[0])
        proposed_energy = self.energy_calculation(proposed_state, proposed_momentum)
        a = 2.0 * ((proposed_energy - current_energy) < -0.5) - 1.0
        while (proposed_energy - current_energy) * a > -a * 0.5:
            self.step_size *= 2.0 ** a
            proposed_state, proposed_momentum = self.leapfrog(self.current_state, p, torch.autograd.grad(current_energy, self.current_state)[0])
            proposed_energy = self.energy_calculation(proposed_state, proposed_momentum)

    def step(self):
        """
        Perform a single NUTS step, using an adaptive algorithm to decide the number of leapfrog steps to take.
        """
        if self.tuning:
            self.find_reasonable_step_size()
            self.tuning = False

        # Initialize momentum randomly from N(0, I)
        p_current = torch.randn_like(self.current_state)
        q_current = self.current_state
        current_energy = self.energy_calculation(q_current, p_current)

        # Build a trajectory and make a decision to stop by the No-U-Turn condition
        q_proposed, p_proposed = q_current, p_current
        for depth in range(self.max_tree_depth):
            q_proposed, p_proposed = self.leapfrog(q_proposed, p_proposed, torch.autograd.grad(current_energy, q_proposed)[0])
            # U-Turn detection: compare angles between initial and current momenta
            if torch.dot(p_current, p_proposed - p_current) <= 0:
                break

        proposed_energy = self.energy_calculation(q_proposed, p_proposed)
        accept_prob = torch.exp(current_energy - proposed_energy)

        if torch.rand(1) < accept_prob:
            self.current_state = q_proposed
            accepted = True
        else:
            accepted = False

        return self.current_state, accepted


class SliceSampler:
    def __init__(self, target_distribution, initial_state, width=1.0, max_steps_out=10):
        """
        Initialize the Slice Sampler with a target distribution, initial state, width of the slice, and max steps out.

        :param target_distribution: A PyTorch distribution object representing the target distribution.
        :param initial_state: A tensor representing the initial state of the Markov chain.
        :param width: The initial guess for the width of the interval.
        :param max_steps_out: The maximum number of steps to expand the interval.
        """
        self.target_distribution = target_distribution
        self.current_state = initial_state
        self.width = width
        self.max_steps_out = max_steps_out

    def find_interval(self, y):
        """
        Find an interval around the current state that contains a significant amount of probability mass at level y.

        :param y: The vertical level defining the slice.
        :return: A tuple of (L, R) defining the left and right bounds of the interval.
        """
        L = self.current_state - self.width * torch.rand(1)  # Randomly position L relative to the current state
        R = L + self.width
        step = 0

        # Step out the interval until the probability at L and R are less than y
        while y < self.target_distribution.log_prob(L) and step < self.max_steps_out:
            L -= self.width
            step += 1
        step = 0
        while y < self.target_distribution.log_prob(R) and step < self.max_steps_out:
            R += self.width
            step += 1

        return L, R

    def sample_within_interval(self, L, R, y):
        """
        Uniformly sample from the slice within the interval [L, R] where the density is above y.

        :param L: The left bound of the interval.
        :param R: The right bound of the interval.
        :param y: The vertical level defining the slice.
        :return: A new state sampled from the interval.
        """
        while True:
            proposed = L + (R - L) * torch.rand(1)  # Uniformly sample a new state within the interval
            if y < self.target_distribution.log_prob(proposed):
                return proposed
            elif proposed < self.current_state:
                L = proposed
            else:
                R = proposed

    def step(self):
        """
        Perform a single slice sampling step.

        :return: The new state of the Markov chain.
        """
        # Draw a vertical level y uniformly from 0 to p(current_state)
        y = self.target_distribution.log_prob(self.current_state) - torch.exp(torch.tensor([1.0]))
        L, R = self.find_interval(y)
        new_state = self.sample_within_interval(L, R, y)
        self.current_state = new_state
        return self.current_state


class GibbsSampler:
    def __init__(self, initial_state):
        """
        Initialize the Gibbs Sampler with an initial state.

        :param initial_state: A tensor representing the initial state of the multivariate Markov chain.
        """
        self.current_state = initial_state

    def conditional_sample(self, index, conditioned_values):
        """
        Sample a variable conditionally based on the values of other variables.

        :param index: The index of the variable to sample.
        :param conditioned_values: The current state of other variables.
        :return: The new value for the variable at the specified index.
        """
        # In a real application, you would define the conditionals specific to your model.
        # Example for a bi-variate normal distribution where sigma1 = sigma2 = 1, rho = 0.8
        mu = torch.tensor([0.0, 0.0])
        rho = 0.8
        sigma = 1.0
        
        if index == 0:
            # Conditional mean of X given Y
            mean = mu[0] + rho / sigma * (conditioned_values[1] - mu[1])
        else:
            # Conditional mean of Y given X
            mean = mu[1] + rho / sigma * (conditioned_values[0] - mu[0])
        
        # Conditional variance
        var = sigma * (1 - rho**2)
        # Return the new sample for the variable
        return torch.normal(mean, var**0.5)

    def step(self):
        """
        Perform one full cycle of Gibbs sampling, updating all variables in the state.

        :return: The updated state of the Markov chain.
        """
        for i in range(len(self.current_state)):
            # Update each variable conditioned on the current values of the others
            self.current_state[i] = self.conditional_sample(i, self.current_state.clone())
        
        return self.current_state


class CustomDistribution:
    def __init__(self, log_prob_func):
        """
        Initialize the Custom Distribution with a log-probability function.

        :param log_prob_func: A function that takes a tensor and returns the log probability density
                              of the tensor under the distribution.
        """
        self.log_prob_func = log_prob_func

    def log_prob(self, x):
        """
        Calculate the log probability of a given state x.

        :param x: A tensor representing the state for which the log probability is calculated.
        :return: The log probability of the state as a tensor.
        """
        if x.requires_grad:
            return self.log_prob_func(x)
        else:
            x = x.clone().detach().requires_grad_(True)
            return self.log_prob_func(x)

    def grad_log_prob(self, x):
        """
        Compute the gradient of the log probability with respect to x.

        :param x: A tensor representing the state for which the gradient of the log probability is computed.
        :return: A tensor representing the gradient of the log probability.
        """
        x = x.clone().detach().requires_grad_(True)
        log_prob = self.log_prob_func(x)
        log_prob.backward()
        return x.grad

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the distribution. Note: This method should be overridden if sampling
        directly from the distribution is required without using MCMC methods.

        :param sample_shape: The shape of the tensor to generate.
        :return: A tensor of samples.
        """
        raise NotImplementedError("Direct sampling method is not implemented.")



class Trace:
    def __init__(self, samples):
        """
        Initialize the Trace class with a set of samples.

        :param samples: A tensor or list of tensors containing the samples from an MCMC run.
        """
        self.samples = torch.stack(samples) if isinstance(samples, list) else samples

    def plot_trace(self):
        """
        Plot the trace of the samples to visualize the Markov Chain behavior over iterations.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.samples.numpy(), label='Trace')
        plt.title('Trace Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Sample Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_autocorrelation(self, max_lag=50):
        """
        Plot the autocorrelation function for the samples to assess the degree of correlation between successive samples.

        :param max_lag: The maximum lag to which the autocorrelation is calculated.
        """
        from statsmodels.graphics.tsaplots import plot_acf
        plt.figure(figsize=(10, 4))
        plot_acf(self.samples.numpy(), lags=max_lag)
        plt.title('Autocorrelation Plot')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

    def compute_effective_sample_size(self):
        """
        Compute the effective sample size (ESS), considering the autocorrelation in the samples.

        :return: An estimate of the effective sample size.
        """
        from statsmodels.tsa.stattools import acf
        autocorr_func = acf(self.samples.numpy(), fft=True, nlags=len(self.samples)//2)
        negative_autocorr = -autocorr_func[1:]
        cum_neg_autocorr = np.cumsum(negative_autocorr)
        positive_autocorr = 1 + 2 * cum_neg_autocorr
        zero_crossings = np.where(np.diff(np.signbit(positive_autocorr)))[0]
        if zero_crossings.size > 0:
            cutoff = zero_crossings[0]
        else:
            cutoff = len(positive_autocorr)
        ess = len(self.samples) / positive_autocorr[cutoff]
        return ess


class Model:
    def __init__(self):
        """
        Initialize the Model class with containers for data, priors, and likelihood.
        """
        self.data = []
        self.priors = {}
        self.likelihood = None
        self.parameters = {}

    def add_data(self, **data):
        """
        Add observational data to the model.

        :param data: Key-value pairs where keys are data identifiers and values are tensors representing the data.
        """
        self.data.append(data)

    def add_prior(self, param_name, distribution):
        """
        Add a prior distribution for a parameter.

        :param param_name: The name of the parameter.
        :param distribution: A PyTorch distribution object representing the prior distribution.
        """
        self.priors[param_name] = distribution
        # Initialize parameters with a sample from the prior
        self.parameters[param_name] = distribution.sample()

    def add_likelihood(self, likelihood_func):
        """
        Set the likelihood function of the model.

        :param likelihood_func: A function that takes parameters and data, and returns the log likelihood.
        """
        self.likelihood = likelihood_func

    def run_sampler(self, sampler_class, num_samples, **sampler_args):
        """
        Run an MCMC sampler to generate posterior samples.

        :param sampler_class: The sampler class to be used (e.g., MetropolisHastingsSampler).
        :param num_samples: Number of samples to generate.
        :param sampler_args: Additional arguments to initialize the sampler.
        :return: A list of samples.
        """
        # Flatten parameters into a single tensor
        initial_params = torch.cat([self.parameters[name] for name in self.parameters])
        
        def target_log_prob(param_tensor):
            log_prob = 0
            offset = 0
            # Decompose the parameter tensor back into parts and calculate their contributions
            for param, prior in self.priors.items():
                param_size = torch.numel(self.parameters[param])
                value = param_tensor[offset:offset+param_size]
                log_prob += prior.log_prob(value).sum()
                offset += param_size
            # Recompose parameters for likelihood if necessary
            param_dict = {param: param_tensor[offset-torch.numel(self.parameters[param]):offset] for param, _ in self.priors.items()}
            if self.likelihood:
                log_prob += self.likelihood(param_dict, self.data)
            return log_prob

        # Custom distribution setup
        class CustomDistribution(torch.distributions.Distribution):
            def log_prob(self, sample):
                return target_log_prob(sample)
        
        # Initialize the sampler
        sampler = sampler_class(target_distribution=CustomDistribution(), initial_state=initial_params, **sampler_args)
        
        # Collect samples
        samples = []
        for _ in range(num_samples):
            sample, _ = sampler.step()
            # If necessary, decompose tensor back into parameter dict
            samples.append({param: sample[offset-torch.numel(self.parameters[param]):offset] for param, offset in zip(self.priors, torch.cumsum(torch.tensor([torch.numel(self.parameters[param]) for param in self.priors]), dim=0))})
        
        return samples


class Visualization:
    def __init__(self, model):
        """
        Initialize the Visualization class with a model which contains priors and parameters.
        
        :param model: A model instance that includes prior distributions and possibly sampled parameters for posteriors.
        """
        self.model = model

    def plot_distribution(self, distribution, label, num_points=100, color='blue'):
        """
        Utility method to plot a distribution provided by a PyTorch distribution object.
        
        :param distribution: The distribution object to plot.
        :param label: Label for the plot.
        :param num_points: Number of points to use in the plot.
        :param color: Color of the plot.
        """
        samples = distribution.sample((num_points,))
        plt.hist(samples.numpy(), bins=30, density=True, alpha=0.6, color=color, label=f'{label} (histogram)')
        plt.title(f'{label} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    def plot_prior(self, param_name):
        """
        Plot the prior distribution for a specified parameter.
        
        :param param_name: The name of the parameter whose prior to plot.
        """
        if param_name in self.model.priors:
            plt.figure(figsize=(8, 4))
            self.plot_distribution(self.model.priors[param_name], f'Prior of {param_name}', color='skyblue')
            plt.show()
        else:
            print(f"No prior defined for parameter {param_name}")

    def plot_posterior(self, param_name, samples):
        """
        Plot the posterior distribution for a specified parameter based on samples.
        
        :param param_name: The name of the parameter whose posterior to plot.
        :param samples: A tensor containing the sampled values for the parameter.
        """
        if param_name in self.model.parameters:
            plt.figure(figsize=(8, 4))
            plt.hist(samples.detach().numpy(), bins=30, density=True, alpha=0.6, color='salmon', label='Posterior (histogram)')
            plt.title(f'Posterior Distribution of {param_name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        else:
            print(f"No samples found for parameter {param_name}")

