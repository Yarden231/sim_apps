import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils import set_rtl

# Call the set_rtl function to apply RTL styles
set_rtl()

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential(lambda_param):
    return -math.log(1 - random.random()) / lambda_param

def sample_normal(mu, sigma):
    u1, u2 = random.random(), random.random()
    return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def sample_composite_distribution():
    return sample_normal(0, 1) if random.random() < 0.2 else sample_normal(3, 1)

def f(x):
    return 3 * x**2

def sample_acceptance_rejection():
    while True:
        x, y = random.random(), random.random() * 3
        if y <= f(x):
            return x

def plot_histogram(samples, title, distribution_func=None, true_density=None):
    fig, ax = plt.subplots(figsize=(6, 3))  # Reduce the size of the plots
    n, bins, _ = ax.hist(samples, bins=30, density=True, alpha=0.7, label='Sampled Data')
    ax.set_title(f"{title} (Number of samples: {len(samples)})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'r-', lw=2, label='True Density Function')
    
    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'g--', lw=2, label='Target Distribution')
    
    ax.legend()
    return fig

def plot_qqplot(samples, title):
    fig, ax = plt.subplots(figsize=(6, 3))  # Reduce the size of the plots
    stats.probplot(samples, dist="norm", plot=ax)
    ax.set_title(f"{title} - QQ Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    return fig

def display_statistics(samples):
    mean = np.mean(samples)
    median = np.median(samples)
    std_dev = np.std(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    st.write(f"**Mean:** {mean:.2f}")
    st.write(f"**Median:** {median:.2f}")
    st.write(f"**Standard Deviation:** {std_dev:.2f}")
    st.write(f"**Minimum Value:** {min_val:.2f}")
    st.write(f"**Maximum Value:** {max_val:.2f}")

def run_sampling(sampling_function, num_samples, update_interval, title, progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples, distribution_func=None, true_density=None):
    samples = []
    for i in range(num_samples):
        samples.append(sampling_function())
        if (i + 1) % update_interval == 0 or i == num_samples - 1:
            # Update histograms and QQ plots side by side
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

            # Update statistics
            stats_placeholder.empty()
            with stats_placeholder:
                display_statistics(samples)
            
            # Print sample values
            if print_samples:
                st.write(f"**Sample values (first {min(10, len(samples))} values):** {samples[:10]}")
        
        progress_bar.progress((i + 1) / num_samples)

def show_sampling_methods():
    st.title("הדגמה של שיטות דגימה שונות")

    st.write("בדף זה נלמד על שיטות דגימה שונות, ונראה דוגמאות כיצד ניתן לייצר דגימות באמצעות Python.")

    if 'selected_sampling' not in st.session_state:
        st.session_state.selected_sampling = None

    num_samples = st.sidebar.slider("מספר דגימות", 100, 10000, 1000)
    update_interval = st.sidebar.slider("תדירות עדכון (מספר דגימות)", 1, 100, 10)

    st.header("בחר שיטת דגימה")

    if st.button("התפלגות אחידה"):
        st.session_state.selected_sampling = 'uniform'
    if st.button("התפלגות מעריכית"):
        st.session_state.selected_sampling = 'exponential'
    if st.button("התפלגות מורכבת"):
        st.session_state.selected_sampling = 'composite'
    if st.button("שיטת הקבלה-דחייה"):
        st.session_state.selected_sampling = 'acceptance_rejection'

    if st.session_state.selected_sampling == 'uniform':
        st.header("1. התפלגות אחידה")
        st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
        a = st.slider("ערך מינימלי (a)", 0.0, 1.0, 0.0)
        b = st.slider("ערך מקסימלי (b)", a + 0.1, 1.0, 1.0)
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: np.ones_like(x) / (b - a)
        run_sampling(lambda: sample_uniform(a, b), num_samples, update_interval, "Uniform Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'exponential':
        st.header("2. התפלגות מעריכית")
        st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
        lambda_param = st.slider("פרמטר למבדא", 0.1, 5.0, 1.0)
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: lambda_param * np.exp(-lambda_param * x)
        run_sampling(lambda: sample_exponential(lambda_param), num_samples, update_interval, "Exponential Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'composite':
        st.header("3. התפלגות מורכבת")
        st.latex(r"f(x) = 0.2 \cdot N(0, 1) + 0.8 \cdot N(3, 1)")
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: 0.2 * stats.norm.pdf(x, 0, 1) + 0.8 * stats.norm.pdf(x, 3, 1)
        run_sampling(sample_composite_distribution, num_samples, update_interval, "Composite Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'acceptance_rejection':
        st.header("4. שיטת הקבלה-דחייה")
        st.latex(r"f(x) = 3x^2, \quad 0 \leq x \leq 1")
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        run_sampling(sample_acceptance_rejection, num_samples, update_interval, "Acceptance-Rejection Method", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, distribution_func=f)

if __name__ == "__main__":
    show_sampling_methods()
