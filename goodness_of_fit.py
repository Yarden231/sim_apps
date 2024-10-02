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

def visualize_samples(samples):
    """Display histograms and QQ plots of the given samples."""
    st.subheader("Histogram and QQ-Plot")

    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(samples, kde=True, ax=ax)
    ax.set_title('Histogram of Samples')
    st.pyplot(fig)

    # QQ-Plot
    fig, ax = plt.subplots()
    stats.probplot(samples, dist="norm", plot=ax)
    ax.set_title('QQ-Plot against Normal Distribution')
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
        # Plot likelihood as a function of μ for fixed σ
        mu_vals = np.linspace(np.mean(samples) - 3, np.mean(samples) + 3, 100)
        sigma = np.std(samples)
        likelihoods = [np.sum(stats.norm.logpdf(samples, loc=mu, scale=sigma)) for mu in mu_vals]

        fig, ax = plt.subplots()
        ax.plot(mu_vals, likelihoods)
        ax.set_title('Log-Likelihood as a function of μ (Normal Distribution)')
        ax.set_xlabel('μ')
        ax.set_ylabel('Log-Likelihood')
        st.pyplot(fig)

    elif distribution == 'Uniform':
        # Plot likelihood as a function of a and b for a uniform distribution
        a_vals = np.linspace(np.min(samples) - 1, np.min(samples) + 1, 100)
        b_vals = np.linspace(np.max(samples) - 1, np.max(samples) + 1, 100)
        likelihoods = np.array([[np.sum(stats.uniform.logpdf(samples, loc=a, scale=b - a))
                                 for b in b_vals] for a in a_vals])

        fig, ax = plt.subplots()
        c = ax.contourf(a_vals, b_vals, likelihoods.T, levels=50, cmap="viridis")
        fig.colorbar(c)
        ax.set_title('Log-Likelihood Contour (Uniform Distribution)')
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        st.pyplot(fig)

def perform_goodness_of_fit(samples, distribution, params):
    """Perform Chi-Square and KS tests."""
    st.subheader("Goodness of Fit Tests")

    # Chi-Square Test
    if distribution == 'Normal':
        observed_freq, bins = np.histogram(samples, bins='auto')
        expected_freq = stats.norm.pdf((bins[:-1] + bins[1:]) / 2, *params) * len(samples)
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

    # Display the samples if they exist
    if 'samples' in st.session_state:
        visualize_samples(st.session_state.samples)

        # Step 2: User selects the distribution they believe the samples come from
        st.subheader("Select the Distribution You Believe the Samples Come From")
        distribution_choice = st.radio("Choose a distribution family", ["Normal", "Exponential", "Uniform"])

        if distribution_choice:
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
