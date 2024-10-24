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
    """Plot histogram with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot histogram
    bins = np.linspace(min(samples), max(samples), 30)
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color='pink', label='Sampled Data')
    
    # Plot true density if provided
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'darkred', linewidth=2, label='True Density')
    
    # Plot target distribution if provided
    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'darkred', linewidth=2, linestyle='--', label='Target Distribution')

    # Styling
    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_qqplot(samples, title):
    """Plot QQ plot with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create QQ plot
    stats.probplot(samples, dist="norm", plot=ax)
    
    # Update colors
    ax.get_lines()[0].set_markerfacecolor('pink')
    ax.get_lines()[0].set_markeredgecolor('darkred')
    ax.get_lines()[1].set_color('darkred')
    
    # Styling
    ax.set_title(f"{title}\nQ-Q Plot")
    ax.grid(True, alpha=0.3)
    
    return fig

def display_statistics(samples):
    """Display statistics with better formatting."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>מדדי מרכז:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>ממוצע: {:.2f} דקות</li>
                    <li>חציון: {:.2f} דקות</li>
                </ul>
            </div>
        """.format(
            np.mean(samples),
            np.median(samples)
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>מדדי פיזור:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>סטיית תקן: {:.2f} דקות</li>
                    <li>טווח: {:.2f} - {:.2f} דקות</li>
                </ul>
            </div>
        """.format(
            np.std(samples),
            np.min(samples),
            np.max(samples)
        ), unsafe_allow_html=True)

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



    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>שיטות דגימה לסימולציית זמני שירות 🚚</h1>
            <p>לאחר שזיהינו את ההתפלגות המתאימה לזמני השירות, נלמד כיצד לייצר דגימות מההתפלגות</p>
        </div>
    """, unsafe_allow_html=True)

    # Normal Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות נורמלית (גאוסיאנית)</h3>
            <p>
                ההתפלגות הנורמלית מתאימה למצבים בהם רוב הערכים מתרכזים סביב הממוצע.
                במשאית המזון, זה מתאים למנות עם זמן הכנה צפוי וסטיות קטנות יחסית.
            </p>
            <ul>
                <li>μ (mu) - הממוצע: מייצג את זמן ההכנה הטיפוסי</li>
                <li>σ (sigma) - סטיית התקן: מייצג את מידת הפיזור סביב הממוצע</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות נורמלית
def sample_normal(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

# דוגמה: דגימת זמני הכנה עם ממוצע 8 דקות וסטיית תקן 1 דקה
samples = np.random.normal(mu=8, sigma=1, size=1000)
```""")

    # Exponential Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות מעריכית</h3>
            <p>
                ההתפלגות המעריכית מתאימה לתיאור זמני המתנה בין אירועים אקראיים,
                או במקרה שלנו - כשיש הרבה זמני הכנה קצרים ומעט זמנים ארוכים.
            </p>
            <ul>
                <li>λ (lambda) - פרמטר הקצב: ככל שהוא גדול יותר, זמני ההכנה קצרים יותר</li>
                <li>הממוצע של ההתפלגות הוא 1/λ</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות מעריכית
def sample_exponential(lambda_param, size):
    return np.random.exponential(scale=1/lambda_param, size=size)

# דוגמה: דגימת זמני הכנה עם ממוצע 5 דקות (λ = 0.2)
samples = np.random.exponential(scale=5, size=1000)
```""")

    # Composite Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות מורכבת (Mixture Distribution)</h3>
            <p>
                התפלגות מורכבת משלבת מספר התפלגויות שונות. במשאית המזון, זה שימושי כאשר:
            </p>
            <ul>
                <li>יש מספר סוגי מנות עם זמני הכנה שונים</li>
                <li>חלק מהמנות פשוטות (זמן קצר) וחלק מורכבות (זמן ארוך)</li>
                <li>יש עומס משתנה בשעות שונות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות מורכבת
def sample_composite(size):
    # 20% מנות פשוטות (ממוצע 5 דקות)
    # 80% מנות מורכבות (ממוצע 10 דקות)
    n_simple = int(0.2 * size)
    n_complex = size - n_simple
    
    # דגימת זמני הכנה למנות פשוטות ומורכבות
    simple_orders = np.random.normal(5, 1, n_simple)
    complex_orders = np.random.normal(10, 1.5, n_complex)
    
    # שילוב הדגימות
    all_orders = np.concatenate([simple_orders, complex_orders])
    
    # וידוא שכל הזמנים חיוביים והגיוניים
    return np.clip(all_orders, 2, 15)

# דוגמה: דגימת 1000 זמני הכנה
samples = sample_composite(1000)
```""")

    # Advanced Example: Data-Driven Distribution
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">דגימה מבוססת נתונים אמיתיים</h3>
            <p>
                לפעמים אנחנו רוצים לדגום מהתפלגות שמבוססת על נתונים אמיתיים שאספנו.
                נוכל להשתמש בשיטת Kernel Density Estimation (KDE) או בדגימה עם החזרה.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
from scipy import stats

def sample_from_data(real_data, size):
    # גישה 1: שימוש ב-KDE
    kde = stats.gaussian_kde(real_data)
    samples_kde = kde.resample(size=size)[0]
    
    # גישה 2: דגימה עם החזרה
    samples_resample = np.random.choice(real_data, size=size, replace=True)
    
    return samples_kde, samples_resample

# דוגמה לשימוש:
real_service_times = np.array([...])  # נתונים אמיתיים שנאספו
kde_samples, resampled = sample_from_data(real_service_times, 1000)
```""")

    # Tips and Best Practices
    st.markdown("""
        <div class="info-box rtl-content">
            <h4>טיפים לדגימה נכונה:</h4>
            <ul>
                <li>תמיד בדקו שהדגימות הגיוניות (למשל, לא יתכנו זמני הכנה שליליים)</li>
                <li>השתמשו ב-np.clip() כדי להגביל את הטווח לערכים הגיוניים</li>
                <li>הוסיפו אקראיות מבוקרת כדי לדמות שונות בזמני ההכנה</li>
                <li>תעדו את הפרמטרים ששימשו לדגימה לצורך שחזור התוצאות</li>
            </ul>
            
            <h4>קוד עזר שימושי:</h4>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# פונקציות עזר שימושיות

def clip_and_validate_times(samples, min_time=2, max_time=15):
    """וידוא שזמני ההכנה הגיוניים"""
    clipped = np.clip(samples, min_time, max_time)
    return clipped

def add_random_variation(samples, variation_percent=10):
    """הוספת שונות אקראית לזמני ההכנה"""
    variation = samples * (variation_percent/100) * np.random.uniform(-1, 1, len(samples))
    return samples + variation

def generate_service_times(distribution_type, size, **params):
    """פונקציה מרכזית לדגימת זמני שירות"""
    if distribution_type == 'normal':
        samples = np.random.normal(params['mu'], params['sigma'], size)
    elif distribution_type == 'exponential':
        samples = np.random.exponential(1/params['lambda'], size)
    elif distribution_type == 'composite':
        samples = sample_composite(size)
    
    # וידוא זמנים הגיוניים והוספת שונות
    samples = clip_and_validate_times(samples)
    samples = add_random_variation(samples)
    
    return samples

# דוגמה לשימוש:
service_times = generate_service_times(
    distribution_type='normal',
    size=1000,
    mu=8,
    sigma=1
)
```""")

    return st.session_state.selected_sampling

def show_sampling_methods():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>שיטות דגימה לסימולציית זמני שירות 🚚</h1>
            <p>לאחר שזיהינו את ההתפלגות המתאימה לזמני השירות, נלמד כיצד לייצר דגימות מההתפלגות</p>
        </div>
    """, unsafe_allow_html=True)

    # Normal Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות נורמלית (גאוסיאנית)</h3>
            <p>
                ההתפלגות הנורמלית מתאימה למצבים בהם רוב הערכים מתרכזים סביב הממוצע.
                במשאית המזון, זה מתאים למנות עם זמן הכנה צפוי וסטיות קטנות יחסית.
            </p>
            <ul>
                <li>μ (mu) - הממוצע: מייצג את זמן ההכנה הטיפוסי</li>
                <li>σ (sigma) - סטיית התקן: מייצג את מידת הפיזור סביב הממוצע</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות נורמלית
def sample_normal(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

# דוגמה: דגימת זמני הכנה עם ממוצע 8 דקות וסטיית תקן 1 דקה
samples = np.random.normal(mu=8, sigma=1, size=1000)
```""")

    # Exponential Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות מעריכית</h3>
            <p>
                ההתפלגות המעריכית מתאימה לתיאור זמני המתנה בין אירועים אקראיים,
                או במקרה שלנו - כשיש הרבה זמני הכנה קצרים ומעט זמנים ארוכים.
            </p>
            <ul>
                <li>λ (lambda) - פרמטר הקצב: ככל שהוא גדול יותר, זמני ההכנה קצרים יותר</li>
                <li>הממוצע של ההתפלגות הוא 1/λ</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות מעריכית
def sample_exponential(lambda_param, size):
    return np.random.exponential(scale=1/lambda_param, size=size)

# דוגמה: דגימת זמני הכנה עם ממוצע 5 דקות (λ = 0.2)
samples = np.random.exponential(scale=5, size=1000)
```""")

    # Composite Distribution Explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התפלגות מורכבת (Mixture Distribution)</h3>
            <p>
                התפלגות מורכבת משלבת מספר התפלגויות שונות. במשאית המזון, זה שימושי כאשר:
            </p>
            <ul>
                <li>יש מספר סוגי מנות עם זמני הכנה שונים</li>
                <li>חלק מהמנות פשוטות (זמן קצר) וחלק מורכבות (זמן ארוך)</li>
                <li>יש עומס משתנה בשעות שונות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# דגימה מהתפלגות מורכבת
def sample_composite(size):
    # 20% מנות פשוטות (ממוצע 5 דקות)
    # 80% מנות מורכבות (ממוצע 10 דקות)
    n_simple = int(0.2 * size)
    n_complex = size - n_simple
    
    # דגימת זמני הכנה למנות פשוטות ומורכבות
    simple_orders = np.random.normal(5, 1, n_simple)
    complex_orders = np.random.normal(10, 1.5, n_complex)
    
    # שילוב הדגימות
    all_orders = np.concatenate([simple_orders, complex_orders])
    
    # וידוא שכל הזמנים חיוביים והגיוניים
    return np.clip(all_orders, 2, 15)

# דוגמה: דגימת 1000 זמני הכנה
samples = sample_composite(1000)
```""")

    # Advanced Example: Data-Driven Distribution
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">דגימה מבוססת נתונים אמיתיים</h3>
            <p>
                לפעמים אנחנו רוצים לדגום מהתפלגות שמבוססת על נתונים אמיתיים שאספנו.
                נוכל להשתמש בשיטת Kernel Density Estimation (KDE) או בדגימה עם החזרה.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
from scipy import stats

def sample_from_data(real_data, size):
    # גישה 1: שימוש ב-KDE
    kde = stats.gaussian_kde(real_data)
    samples_kde = kde.resample(size=size)[0]
    
    # גישה 2: דגימה עם החזרה
    samples_resample = np.random.choice(real_data, size=size, replace=True)
    
    return samples_kde, samples_resample

# דוגמה לשימוש:
real_service_times = np.array([...])  # נתונים אמיתיים שנאספו
kde_samples, resampled = sample_from_data(real_service_times, 1000)
```""")

    # Tips and Best Practices
    st.markdown("""
        <div class="info-box rtl-content">
            <h4>טיפים לדגימה נכונה:</h4>
            <ul>
                <li>תמיד בדקו שהדגימות הגיוניות (למשל, לא יתכנו זמני הכנה שליליים)</li>
                <li>השתמשו ב-np.clip() כדי להגביל את הטווח לערכים הגיוניים</li>
                <li>הוסיפו אקראיות מבוקרת כדי לדמות שונות בזמני ההכנה</li>
                <li>תעדו את הפרמטרים ששימשו לדגימה לצורך שחזור התוצאות</li>
            </ul>
            
            <h4>קוד עזר שימושי:</h4>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""```python
# פונקציות עזר שימושיות

def clip_and_validate_times(samples, min_time=2, max_time=15):
    """וידוא שזמני ההכנה הגיוניים"""
    clipped = np.clip(samples, min_time, max_time)
    return clipped

def add_random_variation(samples, variation_percent=10):
    """הוספת שונות אקראית לזמני ההכנה"""
    variation = samples * (variation_percent/100) * np.random.uniform(-1, 1, len(samples))
    return samples + variation

def generate_service_times(distribution_type, size, **params):
    """פונקציה מרכזית לדגימת זמני שירות"""
    if distribution_type == 'normal':
        samples = np.random.normal(params['mu'], params['sigma'], size)
    elif distribution_type == 'exponential':
        samples = np.random.exponential(1/params['lambda'], size)
    elif distribution_type == 'composite':
        samples = sample_composite(size)
    
    # וידוא זמנים הגיוניים והוספת שונות
    samples = clip_and_validate_times(samples)
    samples = add_random_variation(samples)
    
    return samples

# דוגמה לשימוש:
service_times = generate_service_times(
    distribution_type='normal',
    size=1000,
    mu=8,
    sigma=1
)
```""")

    return st.session_state.selected_sampling

if __name__ == "__main__":
    set_rtl()
    set_ltr_sliders()
    show_sampling_methods()

