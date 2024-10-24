import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from utils import set_rtl, set_ltr_sliders
import pandas as pd

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
    
# Add this function at the beginning of your file, before the visualize_samples_and_qqplots function
def display_samples(samples):
    """Display the first few samples and a simple plot of all samples."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">הצגת המדגם</h3>
            <p>להלן מדגם מייצג של זמני ההכנה שנמדדו:</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display first few samples in a table
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>דוגמאות לזמני הכנה (בדקות):</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a DataFrame with the first 10 samples
        sample_df = pd.DataFrame({
            'מספר מדידה': range(1, 11),
            'זמן (דקות)': samples[:10].round(2)
        }).set_index('מספר מדידה')
        
        st.dataframe(sample_df, height=300)

    with col2:
        # Create a simple line plot of all samples
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(samples, marker='o', linestyle='None', alpha=0.5, markersize=3)
        plt.title('זמני הכנה שנמדדו')
        plt.xlabel('מספר מדידה')
        plt.ylabel('זמן (דקות)')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)


def visualize_samples_and_qqplots(samples):
    """Display histograms and QQ plots of the given samples for three distributions in a grid."""
    st.subheader("Histogram and QQ-Plots for Three Distributions")

    # Create a grid of 2x2 (for histogram and QQ-plots)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Adjust the size to fit the page width

    # Histogram
    sns.histplot(samples, kde=False, ax=axs[0, 0])
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

        # Likelihood as a function of 'a' for a fixed 'b'
        fixed_b = np.max(samples) + 0.5  # Fixing 'b' to a reasonable value
        likelihood_a = [np.sum(stats.uniform.logpdf(samples, loc=a, scale=fixed_b - a)) if fixed_b > a else -np.inf
                        for a in a_vals]

        # Likelihood as a function of 'b' for a fixed 'a'
        fixed_a = np.min(samples) - 0.5  # Fixing 'a' to a reasonable value
        likelihood_b = [np.sum(stats.uniform.logpdf(samples, loc=fixed_a, scale=b - fixed_a)) if b > fixed_a else -np.inf
                        for b in b_vals]

        # Create a grid for the likelihood plots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot likelihood for 'a'
        axs[0].plot(a_vals, likelihood_a)
        axs[0].set_title('Log-Likelihood as a function of a')
        axs[0].set_xlabel('a')
        axs[0].set_ylabel('Log-Likelihood')

        # Plot likelihood for 'b'
        axs[1].plot(b_vals, likelihood_b)
        axs[1].set_title('Log-Likelihood as a function of b')
        axs[1].set_xlabel('b')
        axs[1].set_ylabel('Log-Likelihood')

        st.pyplot(fig)

    elif distribution == 'Exponential':
        # One parameter: λ
        lambda_vals = np.linspace(0.1, 2, 100)
        likelihood_lambda = [np.sum(stats.expon.logpdf(samples, scale=1/lambda_val)) for lambda_val in lambda_vals]

        # Plot the likelihood for λ
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lambda_vals, likelihood_lambda)
        ax.set_title('Log-Likelihood as a function of λ')
        ax.set_xlabel('λ')
        ax.set_ylabel('Log-Likelihood')

        st.pyplot(fig)

# Chi-Square and KS Test for Goodness of Fit
def perform_goodness_of_fit(samples, distribution, params):
    try:
        if distribution == 'Normal':
            # Create the observed frequency from samples
            observed_freq, bins = np.histogram(samples, bins='auto')
            # Calculate midpoints of bins
            bin_midpoints = (bins[:-1] + bins[1:]) / 2

            # Check that params contains both mean (mu) and std (sigma)
            if len(params) != 2:
                raise ValueError("For the Normal distribution, 'params' should be a tuple (mu, sigma).")

            # Generate expected frequencies based on normal distribution and scale them
            expected_freq = stats.norm.pdf(bin_midpoints, loc=params[0], scale=params[1]) * len(samples) * np.diff(bins)

            # Ensure that observed and expected frequencies have the same total sum
            if not np.isclose(observed_freq.sum(), expected_freq.sum(), rtol=1e-8):
                expected_freq *= observed_freq.sum() / expected_freq.sum()

            # Perform the Chi-Square test
            chi_square, p_val_chi = stats.chisquare(observed_freq, expected_freq)
            st.write(f"Chi-Square Test: statistic={chi_square}, p-value={p_val_chi}")

        # KS Test
        if distribution == 'Normal':
            if len(params) != 2:
                raise ValueError("For the Normal distribution, 'params' should be a tuple (mu, sigma).")
            ks_stat, p_val_ks = stats.kstest(samples, 'norm', args=params)

        elif distribution == 'Exponential':
            if len(params) != 1:
                raise ValueError("For the Exponential distribution, 'params' should be a single value (rate).")
            ks_stat, p_val_ks = stats.kstest(samples, 'expon', args=(0, 1 / params[0]))

        elif distribution == 'Uniform':
            if len(params) != 2:
                raise ValueError("For the Uniform distribution, 'params' should be a tuple (a, b).")
            ks_stat, p_val_ks = stats.kstest(samples, 'uniform', args=params)

        st.write(f"KS Test: statistic={ks_stat}, p-value={p_val_ks}")

    except ValueError as e:
        st.error(f"Error during goodness of fit tests: {e}")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from utils import set_rtl, set_ltr_sliders
from styles import get_custom_css

def show():
    set_rtl()
    set_ltr_sliders()
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Header section with story introduction
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>ניתוח זמני שירות - עמדת הכנת המנות 👨‍🍳</h1>
            <p>התאמת מודל סטטיסטי לזמני הכנת מנות במשאית</p>
        </div>
    """, unsafe_allow_html=True)

    # Story introduction
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">רקע</h3>
            <p>
                כדי לשפר את השירות במשאית המזון, ביצענו מדידה של זמני ההכנה בעמדת הבישול.
                העובד בעמדה זו מכין את המנות לפי הזמנת הלקוחות.
                אספנו נתונים של זמני הכנה במשך מספר ימי עבודה, וכעת אנחנו רוצים לנתח את הנתונים
                ולמצוא את ההתפלגות הסטטיסטית שמתארת אותם בצורה הטובה ביותר.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Data generation/loading section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">הנתונים שנאספו</h3>
            <p>להלן מדגם של זמני הכנת המנות (בדקות) שנמדדו בעמדת הבישול:</p>
        </div>
    """, unsafe_allow_html=True)


    # After generating samples but before the QQ plots, add:
    if 'samples' not in st.session_state:
        # Generating realistic cooking time data (between 2 to 15 minutes)
        samples = np.random.lognormal(mean=2, sigma=0.4, size=1000)
        samples = (samples - min(samples)) * (13) / (max(samples) - min(samples)) + 2
        st.session_state.samples = samples

    samples = st.session_state.samples

    # Display summary statistics
    st.markdown("""
        <div class="info-box rtl-content">
            <h4>סטטיסטיקה תיאורית:</h4>
            <ul class="custom-list">
                <li>מספר מדידות: {:d}</li>
                <li>זמן הכנה ממוצע: {:.2f} דקות</li>
                <li>זמן הכנה מינימלי: {:.2f} דקות</li>
                <li>זמן הכנה מקסימלי: {:.2f} דקות</li>
                <li>סטיית תקן: {:.2f} דקות</li>
                <li>חציון: {:.2f} דקות</li>
            </ul>
        </div>
    """.format(
        len(samples),
        np.mean(samples),
        np.min(samples),
        np.max(samples),
        np.std(samples),
        np.median(samples)
    ), unsafe_allow_html=True)

    # Display the raw samples
    display_samples(samples)

    # Add a separator before the QQ plots
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">ניתוח התפלגות הנתונים</h3>
            <p>כעת נבחן את התפלגות הנתונים באמצעות כלים סטטיסטיים:</p>
        </div>
    """, unsafe_allow_html=True)

    # Continue with existing code for QQ plots and rest of the analysis...
    visualize_samples_and_qqplots(samples)
    # Distribution selection
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">בחירת התפלגות מתאימה</h3>
            <p>
                לפי הגרפים שראינו, עלינו לבחור את ההתפלגות שמתאימה ביותר לתיאור זמני ההכנה.
                ההתפלגויות הנפוצות לתיאור זמני שירות הן:
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for the distribution buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות נורמלית</h4>
                <p>מתאימה לזמנים סימטריים סביב הממוצע</p>
            </div>
        """, unsafe_allow_html=True)
        normal_button = st.button("בחר התפלגות נורמלית")

    with col2:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות אחידה</h4>
                <p>מתאימה כשכל הזמנים באותה סבירות</p>
            </div>
        """, unsafe_allow_html=True)
        uniform_button = st.button("בחר התפלגות אחידה")

    with col3:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות מעריכית</h4>
                <p>מתאימה לזמני שירות מוטים</p>
            </div>
        """, unsafe_allow_html=True)
        exp_button = st.button("בחר התפלגות מעריכית")

    # Handle distribution selection
    distribution_choice = None
    if normal_button:
        distribution_choice = 'Normal'
    elif uniform_button:
        distribution_choice = 'Uniform'
    elif exp_button:
        distribution_choice = 'Exponential'

    if distribution_choice:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <p>בחרת את ההתפלגות: {distribution_choice}</p>
            </div>
        """, unsafe_allow_html=True)

        # Maximum Likelihood Estimation
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">אמידת פרמטרים</h3>
                <p>נשתמש בשיטת Maximum Likelihood כדי למצוא את הפרמטרים המתאימים ביותר להתפלגות שנבחרה:</p>
            </div>
        """, unsafe_allow_html=True)
        
        params = estimate_parameters(samples, distribution_choice)
        plot_likelihood(samples, distribution_choice)

        # Goodness of fit tests
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">בדיקת טיב ההתאמה</h3>
                <p>נבצע מבחנים סטטיסטיים כדי לבדוק כמה טוב ההתפלגות שבחרנו מתאימה לנתונים:</p>
            </div>
        """, unsafe_allow_html=True)
        
        perform_goodness_of_fit(samples, distribution_choice, params)

# Keep the existing helper functions (generate_random_samples, visualize_samples_and_qqplots, 
# estimate_parameters, plot_likelihood, perform_goodness_of_fit) as they are
# To show the app, call the show() function
if __name__ == "__main__":
    show()
