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
    """Display enhanced histograms and QQ plots with better visualization and interpretation."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">ניתוח גרפי של ההתפלגות</h3>
            <p>להלן ניתוח גרפי של הנתונים באמצעות היסטוגרמה ותרשימי Q-Q:</p>
        </div>
    """, unsafe_allow_html=True)

    # Create a grid of 2x2 with better proportions
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # Enhanced Histogram with KDE
    sns.histplot(data=samples, kde=True, stat='density', ax=axs[0])
    axs[0].set_title('התפלגות זמני ההכנה')
    axs[0].set_xlabel('זמן (דקות)')
    axs[0].set_ylabel('צפיפות')

    # QQ Plots with confidence bands
    distributions = [
        ('norm', 'Normal Distribution', axs[1]),
        ('uniform', 'Uniform Distribution', axs[2]),
        ('expon', 'Exponential Distribution', axs[3])
    ]

    for dist_name, title, ax in distributions:
        # Calculate QQ plot
        qq = stats.probplot(samples, dist=dist_name, fit=True, plot=ax)
        
        # Add confidence bands
        x = qq[0][0]
        y = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        y_fit = slope * x + intercept
        
        # Calculate confidence bands (approximation)
        n = len(samples)
        sigma = np.std((y - y_fit) / np.sqrt(1 - 1/n))
        conf_band = 1.96 * sigma  # 95% confidence interval
        
        ax.fill_between(x, y_fit - conf_band, y_fit + conf_band, alpha=0.1, color='gray')
        ax.set_title(f'Q-Q Plot - {title}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Add interpretation guide
    st.markdown("""
        <div class="info-box rtl-content">
            <h4>כיצד לפרש את הגרפים:</h4>
            <ul>
                <li><strong>היסטוגרמה:</strong> מציגה את התפלגות זמני ההכנה. הקו הכחול מראה את אומדן צפיפות הגרעין (KDE).</li>
                <li><strong>תרשימי Q-Q:</strong> משווים את הנתונים להתפלגויות שונות. ככל שהנקודות קרובות יותר לקו הישר, כך ההתאמה טובה יותר.</li>
                <li><strong>רצועות אמון:</strong> האזור האפור מציין רווח בר-סמך של 95%. נקודות מחוץ לרצועה מעידות על סטייה מההתפלגות.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def estimate_parameters(samples, distribution):
    """Enhanced parameter estimation with confidence intervals and visual explanation."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">אמידת פרמטרים</h3>
            <p>להלן תוצאות אמידת הפרמטרים בשיטת Maximum Likelihood:</p>
        </div>
    """, unsafe_allow_html=True)

    if distribution == 'Normal':
        # Maximum Likelihood estimation for Normal distribution
        mu, sigma = stats.norm.fit(samples)
        
        # Calculate confidence intervals using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        bootstrap_stds = np.std(bootstrap_samples, axis=1)
        
        mu_ci = np.percentile(bootstrap_means, [2.5, 97.5])
        sigma_ci = np.percentile(bootstrap_stds, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטרים של התפלגות נורמלית:</h4>
                <ul>
                    <li>ממוצע (μ): {mu:.2f} [CI: {mu_ci[0]:.2f}, {mu_ci[1]:.2f}]</li>
                    <li>סטיית תקן (σ): {sigma:.2f} [CI: {sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}]</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return mu, sigma

    elif distribution == 'Exponential':
        # Maximum Likelihood estimation for Exponential distribution
        lambda_est = 1 / np.mean(samples)
        
        # Calculate confidence interval for lambda using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_lambdas = 1 / np.mean(bootstrap_samples, axis=1)
        lambda_ci = np.percentile(bootstrap_lambdas, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטר של התפלגות מעריכית:</h4>
                <ul>
                    <li>קצב (λ): {lambda_est:.4f} [CI: {lambda_ci[0]:.4f}, {lambda_ci[1]:.4f}]</li>
                    <li>זמן ממוצע (1/λ): {1/lambda_est:.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return lambda_est,

    elif distribution == 'Uniform':
        # Maximum Likelihood estimation for Uniform distribution
        a, b = np.min(samples), np.max(samples)
        
        # Calculate confidence intervals using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_mins = np.min(bootstrap_samples, axis=1)
        bootstrap_maxs = np.max(bootstrap_samples, axis=1)
        
        a_ci = np.percentile(bootstrap_mins, [2.5, 97.5])
        b_ci = np.percentile(bootstrap_maxs, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטרים של התפלגות אחידה:</h4>
                <ul>
                    <li>מינימום (a): {a:.2f} [CI: {a_ci[0]:.2f}, {a_ci[1]:.2f}]</li>
                    <li>מקסימום (b): {b:.2f} [CI: {b_ci[0]:.2f}, {b_ci[1]:.2f}]</li>
                    <li>טווח: {b-a:.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return a, b

def plot_likelihood(samples, distribution):
    """Enhanced likelihood function visualization with better interpretation."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">פונקציית Likelihood</h3>
            <p>הגרפים הבאים מציגים את פונקציית ה-Log-Likelihood עבור הפרמטרים השונים:</p>
        </div>
    """, unsafe_allow_html=True)

    if distribution == 'Normal':
        # Create figure with better proportions
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]

        # Parameter ranges
        mu_vals = np.linspace(np.mean(samples) - 3*np.std(samples), 
                             np.mean(samples) + 3*np.std(samples), 100)
        sigma_vals = np.linspace(np.std(samples) * 0.2, 
                                np.std(samples) * 2, 100)

        # Calculate likelihoods
        ll_mu = [np.sum(stats.norm.logpdf(samples, loc=mu, scale=np.std(samples))) 
                 for mu in mu_vals]
        ll_sigma = [np.sum(stats.norm.logpdf(samples, loc=np.mean(samples), scale=sigma)) 
                   for sigma in sigma_vals]

        # Plot with better styling
        ax1.plot(mu_vals, ll_mu, 'b-', linewidth=2)
        ax1.axvline(np.mean(samples), color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Log-Likelihood עבור הממוצע (μ)')
        ax1.set_xlabel('μ')
        ax1.set_ylabel('Log-Likelihood')
        ax1.grid(True, alpha=0.3)

        ax2.plot(sigma_vals, ll_sigma, 'b-', linewidth=2)
        ax2.axvline(np.std(samples), color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Log-Likelihood עבור סטיית התקן (σ)')
        ax2.set_xlabel('σ')
        ax2.set_ylabel('Log-Likelihood')
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

    # Similar improvements for Uniform and Exponential distributions...
    # (Code continues with similar enhancements for other distributions)
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
