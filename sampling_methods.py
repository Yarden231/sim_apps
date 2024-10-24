import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl
from utils import set_ltr_sliders
import time
from styles import get_custom_css

# Call the set_rtl function to apply RTL styles
set_rtl()

def sample_normal(mu, sigma, size):
    """Sample from normal distribution"""
    samples = np.random.normal(mu, sigma, size)
    return np.clip(samples, 2, 15)  # Clip to realistic food prep times

def sample_exponential(lambda_param, size):
    """Sample from exponential distribution"""
    samples = np.random.exponential(1/lambda_param, size)
    return np.clip(samples, 2, 15)  # Clip to realistic food prep times

def sample_composite(size):
    """Sample from mixture of two normal distributions"""
    n_simple = int(0.2 * size)
    n_complex = size - n_simple
    
    simple_orders = np.random.normal(5, 1, n_simple)
    complex_orders = np.random.normal(10, 1.5, n_complex)
    
    all_orders = np.concatenate([simple_orders, complex_orders])
    return np.clip(all_orders, 2, 15)

def plot_histogram(samples, title, distribution_func=None, true_density=None):
    """Plot histogram with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(min(samples), max(samples), 30)
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color='pink', label='Sampled Data')
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'darkred', linewidth=2, label='True Density')

    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'darkred', linewidth=2, linestyle='--', label='Target Distribution')

    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_qqplot(samples, title):
    """Plot QQ plot with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats.probplot(samples, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_markerfacecolor('pink')
    ax.get_lines()[0].set_markeredgecolor('darkred')
    ax.get_lines()[1].set_color('darkred')
    
    ax.set_title(f"{title}\nQ-Q Plot")
    ax.grid(True, alpha=0.3)
    
    return fig

def display_statistics(samples):
    """Display statistics with better formatting."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי מרכז:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>ממוצע: {np.mean(samples):.2f} דקות</li>
                    <li>חציון: {np.median(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי פיזור:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>סטיית תקן: {np.std(samples):.2f} דקות</li>
                    <li>טווח: {np.min(samples):.2f} - {np.max(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def run_sampling(sampling_function, num_samples, update_interval, title, progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=False, distribution_func=None, true_density=None):
    """Run sampling with visualization updates"""
    # Generate all samples at once
    all_samples = sampling_function(num_samples)
    
    # Calculate number of iterations
    num_iterations = (num_samples + update_interval - 1) // update_interval
    
    # Process samples in batches
    samples = []
    for i in range(num_iterations):
        start_idx = i * update_interval
        end_idx = min(start_idx + update_interval, num_samples)
        
        batch_samples = all_samples[start_idx:end_idx]
        samples.extend(batch_samples)
        
        with plot_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_histogram(samples, title, distribution_func, true_density)
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                qqplot_fig = plot_qqplot(samples, title)
                st.pyplot(qqplot_fig)
                plt.close(qqplot_fig)

        stats_placeholder.empty()
        with stats_placeholder:
            display_statistics(samples)
        
        if print_samples:
            st.write(f"**Sample values (first {min(10, len(samples))} values):** {samples[:10]}")
        
        progress = min(1.0, end_idx / num_samples)
        progress_bar.progress(progress)

def show_sampling_methods():
    """Main function to display sampling methods demonstration"""
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>שיטות דגימה לסימולציית זמני שירות 🚚</h1>
            <p>לאחר שזיהינו את ההתפלגות המתאימה לזמני השירות, נלמד כיצד לייצר דגימות מההתפלגות</p>
        </div>
    """, unsafe_allow_html=True)

    # Sample size and update interval selection
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("מספר דגימות", min_value=1000, max_value=10000, value=1000, step=1000)
    with col2:
        update_interval = st.slider("תדירות עדכון", 100, 1000, 100)

    # Distribution selection
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">בחירת התפלגות</h3>
            <p>בחר את סוג ההתפלגות שברצונך לבחון:</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'selected_sampling' not in st.session_state:
        st.session_state.selected_sampling = None

    # Distribution selection buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("התפלגות נורמלית", help="מתאים למנות סטנדרטיות"):
            st.session_state.selected_sampling = 'normal'
    with col2:
        if st.button("התפלגות מעריכית", help="מתאים להזמנות מהירות"):
            st.session_state.selected_sampling = 'exponential'
    with col3:
        if st.button("התפלגות מורכבת", help="מתאים למגוון סוגי מנות"):
            st.session_state.selected_sampling = 'composite'

    # Display selected distribution content
    if st.session_state.selected_sampling == 'normal':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות נורמלית - זמני הכנה למנה סטנדרטית</h3>
                <p>התפלגות זו מתאימה למנות עם זמן הכנה קבוע יחסית.</p>
            </div>
        """, unsafe_allow_html=True)
        
        mu = st.slider("זמן הכנה ממוצע (μ)", 5.0, 15.0, 8.0)
        sigma = st.slider("שונות בזמני ההכנה (σ)", 0.5, 3.0, 1.0)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        true_density = lambda x: stats.norm.pdf(x, mu, sigma)
        run_sampling(
            lambda size: sample_normal(mu, sigma, size),
            num_samples,
            update_interval,
            "Normal Distribution",
            progress_bar,
            plot_placeholder,
            qqplot_placeholder,
            stats_placeholder,
            true_density=true_density
        )

    elif st.session_state.selected_sampling == 'exponential':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות מעריכית - זמני הכנה למנות מהירות</h3>
                <p>התפלגות זו מתאימה למנות שבדרך כלל מוכנות מהר.</p>
            </div>
        """, unsafe_allow_html=True)
        
        lambda_param = st.slider("קצב הכנה (λ)", 0.1, 1.0, 0.5)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        true_density = lambda x: lambda_param * np.exp(-lambda_param * x)
        run_sampling(
            lambda size: sample_exponential(lambda_param, size),
            num_samples,
            update_interval,
            "Exponential Distribution",
            progress_bar,
            plot_placeholder,
            qqplot_placeholder,
            stats_placeholder,
            true_density=true_density
        )

    elif st.session_state.selected_sampling == 'composite':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות מורכבת - זמני הכנה למגוון מנות</h3>
                <p>התפלגות זו מתאימה כאשר יש שני סוגי מנות עיקריים.</p>
            </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        true_density = lambda x: 0.2 * stats.norm.pdf(x, 5, 1) + 0.8 * stats.norm.pdf(x, 10, 1.5)
        run_sampling(
            sample_composite,
            num_samples,
            update_interval,
            "Composite Distribution",
            progress_bar,
            plot_placeholder,
            qqplot_placeholder,
            stats_placeholder,
            true_density=true_density
        )

    # Display code implementation
    if st.session_state.selected_sampling:
        show_implementation()


def show_implementation():
    """Display code implementations with LTR formatting"""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3>מימוש בקוד</h3>
            <p>להלן המימוש של פונקציות הדגימה ב-Python:</p>
        </div>
    """, unsafe_allow_html=True)
    
    code_normal = """
# Sampling from normal distribution
def sample_normal(mu, sigma, size):
    # Generate samples with specified mean and standard deviation
    samples = np.random.normal(mu, sigma, size)
    # Clip values to realistic preparation times (2-15 minutes)
    return np.clip(samples, 2, 15)

# Usage example:
# Sampling preparation times with mean=8 minutes and std=1 minute
samples = sample_normal(mu=8, sigma=1, size=1000)
"""

    code_exponential = """
# Sampling from exponential distribution
def sample_exponential(lambda_param, size):
    # Generate samples with specified rate parameter
    # scale = 1/lambda is the mean time between events
    samples = np.random.exponential(1/lambda_param, size)
    # Clip values to realistic preparation times (2-15 minutes)
    return np.clip(samples, 2, 15)

# Usage example:
# Sampling preparation times with mean=5 minutes (lambda=0.2)
samples = sample_exponential(lambda_param=0.2, size=1000)
"""

    code_composite = """
# Sampling from mixture distribution
def sample_composite(size):
    # Split between simple orders (20%) and complex orders (80%)
    n_simple = int(0.2 * size)
    n_complex = size - n_simple
    
    # Sample preparation times for both types of orders
    # Simple orders: mean=5 minutes, std=1 minute
    simple_orders = np.random.normal(5, 1, n_simple)
    # Complex orders: mean=10 minutes, std=1.5 minutes
    complex_orders = np.random.normal(10, 1.5, n_complex)
    
    # Combine samples from both distributions
    all_orders = np.concatenate([simple_orders, complex_orders])
    # Clip values to realistic preparation times (2-15 minutes)
    return np.clip(all_orders, 2, 15)

# Usage example:
# Sampling 1000 preparation times from mixture distribution
samples = sample_composite(size=1000)
"""

    code_helpers = """
# Utility functions for handling service times

def clip_and_validate_times(samples, min_time=2, max_time=15):
    \"\"\"Ensure service times are within realistic bounds\"\"\"
    return np.clip(samples, min_time, max_time)

def add_random_variation(samples, variation_percent=10):
    \"\"\"Add controlled random variation to service times
    
    Args:
        samples: Array of service times
        variation_percent: Maximum percentage of variation
    
    Returns:
        Array of service times with added random variation
    \"\"\"
    variation = samples * (variation_percent/100) * np.random.uniform(-1, 1, len(samples))
    return samples + variation

def generate_service_times(distribution_type, size, **params):
    \"\"\"Main function for generating service times
    
    Args:
        distribution_type: 'normal', 'exponential', or 'composite'
        size: Number of samples to generate
        **params: Distribution parameters (mu, sigma, lambda)
    
    Returns:
        Array of generated service times
    \"\"\"
    if distribution_type == 'normal':
        samples = np.random.normal(params['mu'], params['sigma'], size)
    elif distribution_type == 'exponential':
        samples = np.random.exponential(1/params['lambda'], size)
    elif distribution_type == 'composite':
        samples = sample_composite(size)
    
    # Validate times and add realistic variation
    samples = clip_and_validate_times(samples)
    samples = add_random_variation(samples)
    
    return samples

# Example usage:
service_times = generate_service_times(
    distribution_type='normal',
    size=1000,
    mu=8,    # mean service time
    sigma=1  # standard deviation
)
"""

    st.markdown('<div dir="ltr">', unsafe_allow_html=True)  # Set LTR direction
    
    with st.expander("Show Implementation Code"):
        if st.session_state.selected_sampling == 'normal':
            st.code(code_normal, language='python')
        elif st.session_state.selected_sampling == 'exponential':
            st.code(code_exponential, language='python')
        elif st.session_state.selected_sampling == 'composite':
            st.code(code_composite, language='python')

        # Add helper functions code
        st.markdown("""
            <div class="custom-card" style="margin-top: 20px;">
                <h4>Helper Functions</h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.code(code_helpers, language='python')

    st.markdown('</div>', unsafe_allow_html=True)  # Close LTR div



if __name__ == "__main__":
    show_sampling_methods()