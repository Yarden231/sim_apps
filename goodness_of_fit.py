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

    # Generate sample data
    if 'samples' not in st.session_state:
        # Generating realistic cooking time data (between 2 to 15 minutes)
        samples = np.random.lognormal(mean=2, sigma=0.4, size=1000)
        samples = (samples - min(samples)) * (13) / (max(samples) - min(samples)) + 2
        st.session_state.samples = samples

    # Display summary statistics
    samples = st.session_state.samples
    st.markdown("""
        <div class="info-box rtl-content">
            <h4>סטטיסטיקה תיאורית:</h4>
            <ul class="custom-list">
                <li>זמן הכנה ממוצע: {:.2f} דקות</li>
                <li>זמן הכנה מינימלי: {:.2f} דקות</li>
                <li>זמן הכנה מקסימלי: {:.2f} דקות</li>
                <li>סטיית תקן: {:.2f} דקות</li>
            </ul>
        </div>
    """.format(
        np.mean(samples),
        np.min(samples),
        np.max(samples),
        np.std(samples)
    ), unsafe_allow_html=True)

    # Visualization of the data
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