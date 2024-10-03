import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def generate_random_samples(sample_size):
    """Generate samples from a random distribution with random parameters."""
    distribution = np.random.choice(['normal', 'uniform', 'exponential'])
    if distribution == 'normal':
        mu = np.random.uniform(-5, 5)
        sigma = np.random.uniform(0.5, 2)
        samples = np.random.normal(loc=mu, scale=sigma, size=sample_size)
        return samples, 'Normal', (mu, sigma)
    elif distribution == 'uniform':
        a = np.random.uniform(-5, 0)
        b = np.random.uniform(0.5, 5)
        samples = np.random.uniform(low=a, high=b, size=sample_size)
        return samples, 'Uniform', (a, b)
    elif distribution == 'exponential':
        lam = np.random.uniform(0.5, 2)
        samples = np.random.exponential(scale=1/lam, size=sample_size)
        return samples, 'Exponential', (lam,)

def visualize_samples_and_qqplots(samples):
    """Display histograms and QQ plots of the given samples for three distributions in a grid."""
    st.subheader("Histogram and QQ-Plots for Three Distributions")

    # Create a grid of 2x2 (for histogram and QQ-plots)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Adjust the size to fit the page width

    # Histogram
    sns.histplot(samples, kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Histogram of Samples')

    # QQ-Plot for Normal Distribution
    stats.probplot(samples, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('QQ-Plot against Normal Distribution')

    # QQ-Plot for Uniform Distribution
    stats.probplot(samples, dist="uniform", plot=axs[1, 0])
    axs[1, 0].set_title('QQ-Plot against Uniform Distribution')

    # QQ-Plot for Exponential Distribution
    stats.probplot(samples, dist="expon", plot=axs[1, 1])
    axs[1, 1].set_title('QQ-Plot against Exponential Distribution')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Display the figure
    st.pyplot(fig)

def estimate_parameters(samples, distribution):
    """Estimate distribution parameters using Maximum Likelihood."""
    if distribution == 'Normal':
        mu, sigma = stats.norm.fit(samples)
        st.write(f"Estimated Parameters for Normal Distribution: μ={mu}, σ={sigma}")
        return mu, sigma
    elif distribution == 'Exponential':
        lambda_est = 1 / np.mean(samples)
        st.write(f"Estimated Parameter for Exponential Distribution: λ={lambda_est}")
        return lambda_est,
    elif distribution == 'Uniform':
        a, b = np.min(samples), np.max(samples)
        st.write(f"Estimated Parameters for Uniform Distribution: a={a}, b={b}")
        return a, b

def plot_likelihood(samples, distribution):
    """Plot the likelihood function based on the sample data."""
    st.subheader("Likelihood Function Plot")

    if distribution == 'Normal':
        # Two parameters: μ and σ
        mu_vals = np.linspace(np.mean(samples) - 3, np.mean(samples) + 3, 100)
        sigma_vals = np.linspace(np.std(samples) * 0.5, np.std(samples) * 1.5, 100)

        # Likelihood as a function of μ for fixed σ
        likelihood_mu = [np.sum(stats.norm.logpdf(samples, loc=mu, scale=np.std(samples))) for mu in mu_vals]
        # Likelihood as a function of σ for fixed μ
        likelihood_sigma = [np.sum(stats.norm.logpdf(samples, loc=np.mean(samples), scale=sigma)) for sigma in sigma_vals]

        # Create a grid for the likelihood plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot likelihood for μ
        axs[0].plot(mu_vals, likelihood_mu)
        axs[0].set_title('Log-Likelihood as a function of μ')
        axs[0].set_xlabel('μ')
        axs[0].set_ylabel('Log-Likelihood')

        # Plot likelihood for σ
        axs[1].plot(sigma_vals, likelihood_sigma)
        axs[1].set_title('Log-Likelihood as a function of σ')
        axs[1].set_xlabel('σ')
        axs[1].set_ylabel('Log-Likelihood')

        st.pyplot(fig)

    elif distribution == 'Uniform':
        # Two parameters: a and b, ensure that a < b
        a_vals = np.linspace(np.min(samples) - 1, np.max(samples) - 0.1, 100)
        b_vals = np.linspace(np.min(samples) + 0.1, np.max(samples) + 1, 100)

        likelihoods = np.array([[np.sum(stats.uniform.logpdf(samples, loc=a, scale=b - a))
                                 if b > a else -np.inf  # Ensuring b > a
                                 for b in b_vals] for a in a_vals])

        # Plot the likelihood as a contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(a_vals, b_vals, likelihoods.T, levels=50, cmap="viridis")
        fig.colorbar(c)
        ax.set_title('Log-Likelihood Contour (Uniform Distribution)')
        ax.set_xlabel('a')
        ax.set_ylabel('b')

        st.pyplot(fig)

    elif distribution == 'Exponential':
        # One parameter: λ
        lambda_vals = np.linspace(0.1, 2, 100)
        likelihood_lambda = [np.sum(stats.expon.logpdf(samples, scale=1/lambda_val)) for lambda_val in lambda_vals]

        # Plot the likelihood for λ
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(lambda_vals, likelihood_lambda)
        ax.set_title('Log-Likelihood as a function of λ')
        ax.set_xlabel('λ')
        ax.set_ylabel('Log-Likelihood')

        st.pyplot(fig)

def perform_goodness_of_fit(samples, distribution, params):
    """Perform Chi-Square and KS tests."""
    st.subheader("Goodness of Fit Tests")

    # Chi-Square Test
    try:
        if distribution == 'Normal':
            # Create the observed frequency from samples
            observed_freq, bins = np.histogram(samples, bins='auto')
            # Calculate midpoints of bins
            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            # Generate expected frequencies based on normal distribution and scale them
            expected_freq = stats.norm.pdf(bin_midpoints, *params) * len(samples) * np.diff(bins)

            # Ensure that observed and expected frequencies have the same total sum
            if not np.isclose(observed_freq.sum(), expected_freq.sum(), rtol=1e-8):
                expected_freq *= observed_freq.sum() / expected_freq.sum()

            # Perform the Chi-Square test
            chi_square, p_val_chi = stats.chisquare(observed_freq, expected_freq)
            st.write(f"Chi-Square Test: statistic={chi_square}, p-value={p_val_chi}")

        # KS Test
        if distribution == 'Normal':
            ks_stat, p_val_ks = stats.kstest(samples, 'norm', args=params)
        elif distribution == 'Exponential':
            ks_stat, p_val_ks = stats.kstest(samples, 'expon', args=(0, 1 / params[0]))
        elif distribution == 'Uniform':
            ks_stat, p_val_ks = stats.kstest(samples, 'uniform', args=params)

        st.write(f"KS Test: statistic={ks_stat}, p-value={p_val_ks}")

    except ValueError as e:
        st.error(f"Error during goodness of fit tests: {e}")

def show():
    """Display the distribution fitting and goodness-of-fit testing page."""
    st.title("Distribution Fitting and Goodness-of-Fit Tests")

    # Add functionality to change sample size
    sample_size = st.slider("Choose Sample Size", min_value=100, max_value=5000, value=1000)

    if st.button("Generate Random Samples"):
        samples, true_distribution, true_params = generate_random_samples(sample_size)
        st.session_state.samples = samples
        st.session_state.true_distribution = true_distribution
        st.session_state.true_params = true_params
        st.write(f"Generated samples from a hidden {true_distribution} distribution.")

    # Display the samples and QQ-plots if they exist
    if 'samples' in st.session_state:
        visualize_samples_and_qqplots(st.session_state.samples)

        # Step 2: User selects the distribution they believe the samples come from
        st.subheader("Select the Distribution You Believe the Samples Come From")

        # Three buttons for user to select the distribution
        if st.button("Normal Distribution"):
            distribution_choice = 'Normal'
        elif st.button("Exponential Distribution"):
            distribution_choice = 'Exponential'
        elif st.button("Uniform Distribution"):
            distribution_choice = 'Uniform'
        else:
            distribution_choice = None

        # If the user has selected a distribution, proceed to the next steps
        if distribution_choice:
            st.write(f"You selected: {distribution_choice}")

            # Step 3: Maximum Likelihood Estimation for chosen distribution
            st.subheader("Maximum Likelihood Estimation")
            params = estimate_parameters(st.session_state.samples, distribution_choice)

            # Plot the likelihood function
            plot_likelihood(st.session_state.samples, distribution_choice)

            # Step 4: Perform goodness-of-fit tests
            perform_goodness_of_fit(st.session_state.samples, distribution_choice, params)

# To show the app, call the show() function
if __name__ == "__main__":
    show()
