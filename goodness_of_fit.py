import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from utils import set_rtl, set_ltr_sliders
from styles import get_custom_css

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
            'Sample #': range(1, 11),
            'Time (minutes)': samples[:10].round(2)
        }).set_index('Sample #')
        
        st.dataframe(sample_df, height=300)

    with col2:
        # Create a simple line plot of all samples
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(samples, marker='o', linestyle='None', alpha=0.5, markersize=3)
        plt.title('Service Times')
        plt.xlabel('Sample Number')
        plt.ylabel('Time (minutes)')
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

def perform_goodness_of_fit(samples, distribution, params):
    """
    Improved goodness of fit testing with proper handling of different distributions
    and degrees of freedom.
    """
    test_results = []
    
    # Calculate number of bins using Sturges' rule
    n_bins = int(np.ceil(np.log2(len(samples)) + 1))
    
    # Perform Chi-Square Test
    observed_freq, bins = np.histogram(samples, bins=n_bins)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    # Calculate expected frequencies based on the distribution
    if distribution == 'Normal':
        mu, sigma = params
        expected_probs = stats.norm.cdf(bins[1:], mu, sigma) - stats.norm.cdf(bins[:-1], mu, sigma)
        dof = len(observed_freq) - 3  # Subtract 3 for normal (2 parameters + 1)
        
    elif distribution == 'Exponential':
        lambda_param = params[0]
        expected_probs = stats.expon.cdf(bins[1:], scale=1/lambda_param) - stats.expon.cdf(bins[:-1], scale=1/lambda_param)
        dof = len(observed_freq) - 2  # Subtract 2 for exponential (1 parameter + 1)
        
    elif distribution == 'Uniform':
        a, b = params
        expected_probs = stats.uniform.cdf(bins[1:], a, b-a) - stats.uniform.cdf(bins[:-1], a, b-a)
        dof = len(observed_freq) - 3  # Subtract 3 for uniform (2 parameters + 1)
    
    expected_freq = expected_probs * len(samples)
    
    # Remove bins with expected frequency < 5 (combining them)
    mask = expected_freq >= 5
    if not all(mask):
        observed_freq = np.array([sum(observed_freq[~mask])] + list(observed_freq[mask]))
        expected_freq = np.array([sum(expected_freq[~mask])] + list(expected_freq[mask]))
        dof -= len(mask) - sum(mask) - 1
    
    # Perform Chi-Square test with correct degrees of freedom
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    p_value_chi = 1 - stats.chi2.cdf(chi_square_stat, dof)
    
    test_results.append(f"Chi-Square Test: statistic={chi_square_stat:.4f}, p-value={p_value_chi:.4f}")
    
    # Perform Kolmogorov-Smirnov test
    if distribution == 'Normal':
        ks_stat, p_value_ks = stats.kstest(samples, 'norm', args=params)
    elif distribution == 'Exponential':
        ks_stat, p_value_ks = stats.kstest(samples, 'expon', args=(0, 1/params[0]))
    elif distribution == 'Uniform':
        ks_stat, p_value_ks = stats.kstest(samples, 'uniform', args=params)
    
    test_results.append(f"KS Test: statistic={ks_stat:.4f}, p-value={p_value_ks:.4f}")
    
    # Format results for display
    conclusion = "מסקנות המבחנים הסטטיסטיים:\n\n"
    
    for result in test_results:
        test_name = result.split(':')[0]
        p_value = float(result.split('p-value=')[1])
        
        if p_value < 0.05:
            conclusion += f"• {test_name}: דוחים את השערת האפס (H0). הנתונים כנראה אינם מתפלגים לפי ההתפלגות הנבחרת.\n"
        else:
            conclusion += f"• {test_name}: אין מספיק עדות לדחות את השערת האפס (H0). ייתכן שההתפלגות מתאימה לנתונים.\n"
    
    return test_results, conclusion

def generate_service_times(size=1000, distribution_type=None):
    """
    Generate service times from various distributions.
    If distribution_type is None, randomly select one.
    """
    # Use numpy random seed based on current timestamp
    np.random.seed(int(pd.Timestamp.now().timestamp()))
    
    if distribution_type is None:
        distribution_type = np.random.choice([
            'normal', 'uniform', 'exponential', 'mixture', 'lognormal'
        ])
    
    def scale_times(times, min_time=2, max_time=15):
        """Scale times to be between min_time and max_time minutes"""
        return (times - np.min(times)) * (max_time - min_time) / (np.max(times) - np.min(times)) + min_time
    
    if distribution_type == 'normal':
        # Normal distribution with realistic parameters
        mu = np.random.uniform(7, 9)  # mean service time
        sigma = np.random.uniform(1, 2)  # standard deviation
        samples = np.random.normal(mu, sigma, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Normal', 'params': {'mu': mu, 'sigma': sigma}}
        
    elif distribution_type == 'uniform':
        # Uniform distribution between min and max times
        a = np.random.uniform(2, 5)  # minimum time
        b = np.random.uniform(10, 15)  # maximum time
        samples = np.random.uniform(a, b, size)
        dist_info = {'type': 'Uniform', 'params': {'a': a, 'b': b}}
        
    elif distribution_type == 'exponential':
        # Exponential distribution scaled to realistic times
        lambda_param = np.random.uniform(0.15, 0.25)  # rate parameter
        samples = np.random.exponential(1/lambda_param, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Exponential', 'params': {'lambda': lambda_param}}
        
    elif distribution_type == 'lognormal':
        # Lognormal distribution for right-skewed times
        mu = np.random.uniform(1.8, 2.2)
        sigma = np.random.uniform(0.2, 0.4)
        samples = np.random.lognormal(mu, sigma, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Lognormal', 'params': {'mu': mu, 'sigma': sigma}}
        
    elif distribution_type == 'mixture':
        # Mixture of distributions
        mixture_type = np.random.choice([
            'normal_exponential',
            'normal_uniform',
            'bimodal_normal'
        ])
        
        if mixture_type == 'normal_exponential':
            # Mix of normal (regular orders) and exponential (rush orders)
            prop_normal = np.random.uniform(0.6, 0.8)
            n_normal = int(size * prop_normal)
            n_exp = size - n_normal
            
            normal_samples = np.random.normal(8, 1.5, n_normal)
            exp_samples = np.random.exponential(2, n_exp) + 5
            samples = np.concatenate([normal_samples, exp_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Normal-Exponential',
                'params': {'proportion_normal': prop_normal}
            }
            
        elif mixture_type == 'normal_uniform':
            # Mix of normal (regular orders) and uniform (special orders)
            prop_normal = np.random.uniform(0.7, 0.9)
            n_normal = int(size * prop_normal)
            n_uniform = size - n_normal
            
            normal_samples = np.random.normal(8, 1.5, n_normal)
            uniform_samples = np.random.uniform(4, 12, n_uniform)
            samples = np.concatenate([normal_samples, uniform_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Normal-Uniform',
                'params': {'proportion_normal': prop_normal}
            }
            
        else:  # bimodal_normal
            # Bimodal normal for different types of orders
            prop_fast = np.random.uniform(0.5, 0.7)
            n_fast = int(size * prop_fast)
            n_slow = size - n_fast
            
            fast_samples = np.random.normal(6, 1, n_fast)
            slow_samples = np.random.normal(11, 1.5, n_slow)
            samples = np.concatenate([fast_samples, slow_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Bimodal-Normal',
                'params': {'proportion_fast': prop_fast}
            }
        
        samples = scale_times(samples)
    
    # Ensure all times are positive and within realistic bounds
    samples = np.clip(samples, 2, 15)
    
    return samples, dist_info

def show():
    set_rtl()
    set_ltr_sliders()
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Header section
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>ניתוח זמני שירות - עמדת הכנת המנות 👨‍🍳</h1>
            <p>התאמת מודל סטטיסטי לזמני הכנת מנות במשאית</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Generate new samples
    if 'samples' not in st.session_state or st.button('יצירת מדגם חדש'):
        samples, dist_info = generate_service_times()
        st.session_state.samples = samples
        st.session_state.dist_info = dist_info
        
        # Add this for debugging/testing
        st.markdown(f"""
            <div class="info-box rtl-content">
                <p>התפלגות אמיתית (למטרות בדיקה): {dist_info['type']}</p>
                {'<p>תת-סוג: ' + dist_info.get('subtype', 'N/A') + '</p>' if 'subtype' in dist_info else ''}
                <p>פרמטרים: {dist_info['params']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display summary statistics
    samples = st.session_state.samples
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
    
    # Continue with the rest of your display_samples(), visualize_samples_and_qqplots(),
    # and other visualization functions...

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

# To show the app, call the show() function
if __name__ == "__main__":
    show()
