import streamlit as st
import plotly.graph_objects as go
import numpy as np

def show():
    st.title("תיאוריה של סימולציה")

    st.write("""
    בדף זה תמצאו סקירה של הנושאים התיאורטיים העיקריים בקורס הסימולציה. 
    לחצו על כל נושא כדי להרחיב ולקבל מידע נוסף.
    """)

    topics = {
        "מבוא לסימולציה": intro_to_simulation,
        "מודלים של סימולציה": simulation_models,
        "יצירת מספרים אקראיים": random_number_generation,
        "ניתוח תוצאות סימולציה": simulation_results_analysis,
        "תכנון ניסויי סימולציה": simulation_experiment_design
    }

    for topic, function in topics.items():
        with st.expander(topic):
            function()

def intro_to_simulation():
    st.subheader("מבוא לסימולציה")
    st.write("""
    סימולציה היא טכניקה לחיקוי התנהגות של מערכת או תהליך בעולם האמיתי. 
    היא משמשת לניתוח, תכנון ושיפור מערכות מורכבות בתחומים רבים כמו הנדסה, כלכלה, רפואה ועוד.
    
    יתרונות הסימולציה:
    - מאפשרת לבחון תרחישים שונים ללא סיכון או עלות גבוהה
    - מספקת תובנות על התנהגות מערכות מורכבות
    - מאפשרת לבחון השפעות של שינויים במערכת לפני יישומם בפועל
    """)

def simulation_models():
    st.subheader("מודלים של סימולציה")
    st.write("""
    קיימים מספר סוגים של מודלי סימולציה:
    
    1. סימולציה בדידה
    2. סימולציה רציפה
    3. סימולציה מבוססת סוכנים
    4. סימולציית מונטה קרלו
    
    כל סוג מתאים לסוגים שונים של בעיות ומערכות.
    """)
    
    st.code("""
    # דוגמה פשוטה לסימולציית מונטה קרלו בפייתון
    import random

    def monte_carlo_pi(n):
        inside_circle = 0
        total_points = n
        
        for _ in range(total_points):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x*x + y*y <= 1:
                inside_circle += 1
        
        pi_estimate = 4 * inside_circle / total_points
        return pi_estimate

    # הרצת הסימולציה
    estimated_pi = monte_carlo_pi(1000000)
    print(f"הערכת π: {estimated_pi}")
    """, language="python")

def random_number_generation():
    st.subheader("יצירת מספרים אקראיים")
    st.write("""
    יצירת מספרים אקראיים היא מרכיב קריטי בסימולציה. 
    בפועל, מחשבים משתמשים במספרים פסאודו-אקראיים, שנוצרים באמצעות אלגוריתמים דטרמיניסטיים.
    
    שיטות נפוצות:
    - שיטת הקונגרואנציה הלינארית
    - מחוללי Mersenne Twister
    - מחוללים קריפטוגרפיים
    """)
    
    # יצירת היסטוגרמה של מספרים אקראיים
    random_numbers = np.random.rand(1000)
    fig = go.Figure(data=[go.Histogram(x=random_numbers)])
    fig.update_layout(title_text='התפלגות של 1000 מספרים אקראיים', xaxis_title_text='ערך', yaxis_title_text='תדירות')
    st.plotly_chart(fig)

def simulation_results_analysis():
    st.subheader("ניתוח תוצאות סימולציה")
    st.write("""
    ניתוח תוצאות הסימולציה כולל:
    
    1. ניתוח סטטיסטי של הנתונים
    2. בדיקת מובהקות סטטיסטית
    3. ניתוח רגישות
    4. הסקת מסקנות והמלצות
    
    חשוב לזכור שתוצאות הסימולציה הן הערכות, ויש להתייחס אליהן בהתאם.
    """)

def simulation_experiment_design():
    st.subheader("תכנון ניסויי סימולציה")
    st.write("""
    תכנון נכון של ניסויי סימולציה חיוני להשגת תוצאות אמינות ומועילות. שלבי התכנון כוללים:
    
    1. הגדרת מטרות הניסוי
    2. בחירת הפרמטרים והמשתנים לבדיקה
    3. קביעת מספר הריצות והזמן לכל ריצה
    4. תכנון שיטות לאיסוף וניתוח הנתונים
    
    תכנון טוב יבטיח שהסימולציה תספק תשובות לשאלות הרלוונטיות ותמקסם את התועלת מהניסוי.
    """)

    st.markdown("""
    לקריאה נוספת על תכנון ניסויי סימולציה, בקרו באתר:
    [תכנון ניסויים בסימולציה](https://www.simulationscience.org/tutorials/experimental-design-simulation-studies)
    """)

# אין צורך לקרוא לפונקציה show_theory() כאן, היא תיקרא מ-main.py
