import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    fig, ax = plt.subplots()
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

def run_sampling(sampling_function, num_samples, update_interval, title, progress_bar, plot_placeholder, distribution_func=None, true_density=None):
    samples = []
    for i in range(num_samples):
        samples.append(sampling_function())
        if (i + 1) % update_interval == 0 or i == num_samples - 1:
            fig = plot_histogram(samples, title, distribution_func, true_density)
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        progress_bar.progress((i + 1) / num_samples)

def show_sampling_methods():
    st.title("הדגמה של שיטות דגימה שונות")

    num_samples = st.sidebar.slider("מספר דגימות", 100, 10000, 1000)
    update_interval = st.sidebar.slider("תדירות עדכון (מספר דגימות)", 1, 100, 10)

    st.header("1. התפלגות אחידה")
    st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
    a = st.slider("ערך מינימלי (a)", 0.0, 1.0, 0.0)
    b = st.slider("ערך מקסימלי (b)", a + 0.1, 1.0, 1.0)
    
    st.code("""
    def sample_uniform(a, b):
        return a + (b - a) * random.random()
    """)
    
    if st.button("התחל דגימה מהתפלגות אחידה"):
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        true_density = lambda x: np.ones_like(x) / (b - a)
        run_sampling(lambda: sample_uniform(a, b), num_samples, update_interval, "Uniform Distribution", progress_bar, plot_placeholder, true_density=true_density)

    st.header("2. התפלגות מעריכית")
    st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
    lambda_param = st.slider("פרמטר למבדא", 0.1, 5.0, 1.0)
    
    st.code("""
    def sample_exponential(lambda_param):
        return -math.log(1 - random.random()) / lambda_param
    """)
    
    if st.button("התחל דגימה מהתפלגות מעריכית"):
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        true_density = lambda x: lambda_param * np.exp(-lambda_param * x)
        run_sampling(lambda: sample_exponential(lambda_param), num_samples, update_interval, "Exponential Distribution", progress_bar, plot_placeholder, true_density=true_density)

    st.header("3. התפלגות מורכבת")
    st.latex(r"f(x) = 0.2 \cdot N(0, 1) + 0.8 \cdot N(3, 1)")
    
    st.code("""
    def sample_composite_distribution():
        return sample_normal(0, 1) if random.random() < 0.2 else sample_normal(3, 1)
    """)
    
    if st.button("התחל דגימה מהתפלגות מורכבת"):
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        true_density = lambda x: 0.2 * stats.norm.pdf(x, 0, 1) + 0.8 * stats.norm.pdf(x, 3, 1)
        run_sampling(sample_composite_distribution, num_samples, update_interval, "Composite Distribution", progress_bar, plot_placeholder, true_density=true_density)

    st.header("4. שיטת הקבלה-דחייה")
    st.latex(r"f(x) = 3x^2, \quad 0 \leq x \leq 1")
    
    st.code("""
    def f(x):
        return 3 * x**2

    def sample_acceptance_rejection():
        while True:
            x, y = random.random(), random.random() * 3
            if y <= f(x):
                return x
    """)
    
    if st.button("התחל דגימה בשיטת הקבלה-דחייה"):
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        run_sampling(sample_acceptance_rejection, num_samples, update_interval, "Acceptance-Rejection Method", progress_bar, plot_placeholder, distribution_func=f)

if __name__ == "__main__":
    show_sampling_methods()