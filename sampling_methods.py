import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl
from utils import set_ltr_sliders
import time
# Call the set_rtl function to apply RTL styles
set_rtl()

def sample_uniform(a, b, size):
    return np.random.uniform(a, b, size)

def sample_exponential(lambda_param, size):
    return np.random.exponential(1/lambda_param, size)

def sample_normal(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

def sample_composite_distribution(size):
    normal_1 = np.random.normal(0, 1, size)
    normal_2 = np.random.normal(3, 1, size)
    mask = np.random.rand(size) < 0.2
    return np.where(mask, normal_1, normal_2)

def sample_acceptance_rejection(size):
    samples = []
    while len(samples) < size:
        x = np.random.random()
        y = np.random.random() * 3
        if y <= f(x):
            samples.append(x)
    return np.array(samples)

def f(x):
    return 3 * x ** 2

def plot_histogram(samples, title, distribution_func=None, true_density=None):
    fig, ax = plt.subplots(figsize=(6, 4))  # Fixed figure size
    ax.hist(samples, bins=100, density=True, alpha=0.7, label='Sampled Data')
    ax.set_title(f"{title} (Number of samples: {len(samples)})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'r-', lw=2, label='True Density Function')
    
    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'g--', lw=2, label='Target Distribution')

    ax.legend(loc='upper right')  # Fixed legend location
    ax.set_xlim([min(samples), max(samples)])  # Set axis limits
    ax.set_ylim(0, 2.0)  # Fixed y-axis limit for consistency
    ax.grid(True)  # Add grid for clarity
    return fig

def plot_qqplot(samples, title):
    fig, ax = plt.subplots(figsize=(6, 4))  # Fixed figure size
    stats.probplot(samples, dist="norm", plot=ax)
    ax.set_title(f"{title} - QQ Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True)  # Add grid for clarity
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
    # Generate all samples at once
    all_samples = sampling_function(num_samples)
    
    # Simulate real-time updates by splitting samples into batches
    samples = []
    for i in range(0, num_samples, update_interval):
        batch_samples = all_samples[i:i+update_interval]
        samples.extend(batch_samples)
        
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
        
        # Simulate progress in real-time
        progress_bar.progress((i + update_interval) / num_samples)
        
        # Delay to simulate real-time sampling (optional)
        #time.sleep(0.01)

def show_sampling_methods():
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>שיטות דגימה לסימולציית זמני שירות 🚚</h1>
            <p>לאחר שזיהינו את ההתפלגות המתאימה לזמני השירות, נלמד כיצד לייצר דגימות מההתפלגות</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">רקע</h3>
            <p>
                כדי לבנות סימולציה של משאית המזון, עלינו לדעת כיצד לייצר זמני שירות מלאכותיים שמתנהגים
                כמו הזמנים האמיתיים. נכיר מספר שיטות דגימה שיעזרו לנו לייצר זמנים אלו:
            </p>
            <ul>
                <li><strong>התפלגות אחידה:</strong> מתאימה למצבים בהם זמן ההכנה הוא אקראי לחלוטין בין מינימום למקסימום</li>
                <li><strong>התפלגות נורמלית:</strong> מתאימה כאשר זמן ההכנה מתרכז סביב ממוצע עם סטיות סימטריות</li>
                <li><strong>התפלגות מעריכית:</strong> מתאימה למצבים בהם יש הרבה זמני הכנה קצרים ומעט זמנים ארוכים</li>
                <li><strong>התפלגות מורכבת:</strong> מתאימה כאשר יש מספר סוגי הזמנות עם זמני הכנה שונים</li>
                <li><strong>שיטת קבלה-דחייה:</strong> מתאימה כאשר התפלגות זמני ההכנה היא ייחודית</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Session state initialization
    if 'selected_sampling' not in st.session_state:
        st.session_state.selected_sampling = None

    # Parameters selection
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">הגדרת פרמטרים</h3>
            <p>בחר את מספר הדגימות ותדירות העדכון:</p>
        </div>
    """, unsafe_allow_html=True)

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

    # Distribution specific interfaces
    if st.session_state.selected_sampling == 'normal':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות נורמלית - זמני הכנה למנה סטנדרטית</h3>
                <p>
                    התפלגות זו מתאימה למנות עם זמן הכנה קבוע יחסית. 
                    הממוצע (μ) מייצג את זמן ההכנה הטיפוסי, וסטיית התקן (σ) מייצגת את מידת השונות בזמנים.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("זמן הכנה ממוצע (μ)", 5.0, 15.0, 8.0)
        with col2:
            sigma = st.slider("שונות בזמני ההכנה (σ)", 0.5, 3.0, 1.0)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: stats.norm.pdf(x, mu, sigma)
        run_sampling(lambda size: sample_normal(mu, sigma, size), num_samples, update_interval, 
                    "התפלגות זמני הכנה למנה סטנדרטית", progress_bar, plot_placeholder, 
                    qqplot_placeholder, stats_placeholder, print_samples=False, true_density=true_density)

    elif st.session_state.selected_sampling == 'exponential':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות מעריכית - זמני הכנה למנות מהירות</h3>
                <p>
                    התפלגות זו מתאימה למנות שבדרך כלל מוכנות מהר, אך לעתים לוקחות זמן רב יותר.
                    הפרמטר λ קובע את הקצב הממוצע - ככל שהוא גדול יותר, זמני ההכנה קצרים יותר.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        lambda_param = st.slider("קצב הכנה (λ)", 0.1, 1.0, 0.5, 
                               help="ערך גבוה יותר = זמני הכנה קצרים יותר בממוצע")
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: lambda_param * np.exp(-lambda_param * x)
        run_sampling(lambda size: sample_exponential(lambda_param, size), num_samples, update_interval,
                    "התפלגות זמני הכנה למנות מהירות", progress_bar, plot_placeholder,
                    qqplot_placeholder, stats_placeholder, print_samples=False, true_density=true_density)

    elif st.session_state.selected_sampling == 'composite':
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>התפלגות מורכבת - זמני הכנה למגוון מנות</h3>
                <p>
                    התפלגות זו מתאימה כאשר יש שני סוגי מנות עיקריים:
                    מנות פשוטות שמוכנות מהר (כ-20% מההזמנות) ומנות מורכבות שלוקחות יותר זמן (כ-80% מההזמנות).
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: 0.2 * stats.norm.pdf(x, 5, 1) + 0.8 * stats.norm.pdf(x, 10, 1.5)
        
        def modified_composite_distribution(size):
            # Modified to give more realistic food preparation times
            simple_orders = np.random.normal(5, 1, int(0.2 * size))  # Simple orders: ~5 minutes
            complex_orders = np.random.normal(10, 1.5, size - len(simple_orders))  # Complex orders: ~10 minutes
            all_orders = np.concatenate([simple_orders, complex_orders])
            return np.clip(all_orders, 2, 15)  # Ensure times are between 2 and 15 minutes
        
        run_sampling(modified_composite_distribution, num_samples, update_interval,
                    "התפלגות זמני הכנה למגוון מנות", progress_bar, plot_placeholder,
                    qqplot_placeholder, stats_placeholder, print_samples=False, true_density=true_density)

    # Add explanation of plots
    if st.session_state.selected_sampling:
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>הסבר על הגרפים:</h4>
                <ul>
                    <li><strong>היסטוגרמה:</strong> מציגה את התפלגות זמני ההכנה שנדגמו. הקו האדום מראה את ההתפלגות התיאורטית.</li>
                    <li><strong>תרשים Q-Q:</strong> משמש לבדיקת ההתאמה להתפלגות הנבחרת. ככל שהנקודות קרובות יותר לקו, ההתאמה טובה יותר.</li>
                    <li><strong>סטטיסטיקה תיאורית:</strong> מציגה מדדים סטטיסטיים בסיסיים של זמני ההכנה שנדגמו.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    set_rtl()
    set_ltr_sliders()
    show_sampling_methods()

