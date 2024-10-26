# home.py
import streamlit as st
import matplotlib.pyplot as plt
from styles import get_custom_css

def create_order_dynamics_section():
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">2. דינמיקת עמדת ההזמנות 📝</h3>
            <p>לקוחות עם דחיפויות והעדפות שונות משפיעים על זמני עיבוד ההזמנות:</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='text-align: center; font-weight: bold;'>סוג לקוח</div>", unsafe_allow_html=True)
        st.markdown("""
        A סוג<br>
        B סוג<br>
        C סוג
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align: center; font-weight: bold;'>אחוז מהלקוחות</div>", unsafe_allow_html=True)
        st.markdown("""
        50%<br>
        25%<br>
        25%
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='text-align: center; font-weight: bold;'>זמן הזמנה</div>", unsafe_allow_html=True)
        st.markdown("""
        אחיד (3-4 דקות) - המהיר ביותר<br>
        משולש (4-6 דקות) - בינוני<br>
        קבוע (10 דקות) - האיטי ביותר
        """, unsafe_allow_html=True)

def create_meal_prep_section():
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">3. פרטי הכנת הארוחות 👨‍🍳</h3>
            
            <div class="info-box">
                <h4>זמני בישול סטוכסטיים:</h4>
                <ul class="custom-list">
                    <li>התפלגות נורמלית</li>
                    <li>ממוצע (μ): 5 דקות - הזמן הממוצע להכנת מנה</li>
                    <li>סטיית תקן (σ): דקה אחת - שונות בזמני ההכנה</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: right;'>הסתברויות לפי גודל המנה:</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div style='text-align: center; font-weight: bold;'>גודל מנה</div>", unsafe_allow_html=True)
        st.markdown("""
        מנה בודדת<br>
        זוג מנות<br>
        שלוש מנות
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align: center; font-weight: bold;'>הסתברות</div>", unsafe_allow_html=True)
        st.markdown("""
        20%<br>
        50%<br>
        30%
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='text-align: center; font-weight: bold;'>השפעה</div>", unsafe_allow_html=True)
        st.markdown("""
        שירות מהיר ואישי<br>
        איזון בין מהירות ואיכות<br>
        יעילות גבוהה
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div style='text-align: center; font-weight: bold;'>סיכוי לבישול חסר</div>", unsafe_allow_html=True)
        st.markdown("""
        0%<br>
        0%<br>
        30%
        """, unsafe_allow_html=True)

def show():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>🚚 משאית האוכל העמוסה</h1>
            <p>סימולציה אינטראקטיבית המבוססת על תורת התורים והתפלגויות הסתברותיות, המדמה את האתגרים של ניהול משאית אוכל רחוב</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">ברוכים הבאים!</h3>
                <p>
                    צאו למסע מרתק אל תוך פעילות היומיום של משאית האוכל העמוסה. 
                    בדוגמה זו נחקור את האתגרים היומיומיים של משאית אוכל מבוקשת, 
                    בה מתחלפים לקוחות, נרשמות הזמנות ומתבצעת הכנה מרובת שלבים.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        try:
            image = plt.imread("food_track_image.jpg")
            st.image(image, use_column_width=True, caption="משאית המזון שלנו")
        except:
            st.info("תמונת משאית המזון לא נמצאה. אנא וודאו שהקובץ 'food_track_image.jpg' קיים בתיקייה.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer Arrival section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">1. הגעת לקוחות 👥</h3>
            <ul class="custom-list">
                <li>הגעות אקראיות: לקוחות מגיעים באופן ספונטני, נמשכים לתפריט המפתה</li>
                <li>התפלגות מעריכית: מרווחי ההגעה ממוצעים כ-6 דקות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Order Dynamics and Meal Prep sections using the functions
    create_order_dynamics_section()
    create_meal_prep_section()
    
    # Pickup Time section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">4. זמן איסוף 🕒</h3>
            <div class="metric-container">
                <p>התפלגות אחידה: זמני האיסוף נעים בין 2 ל-4 דקות</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Customer Patience section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">5. סבלנות הלקוחות 😊</h3>
            <div class="info-box">
                <p>הסתברות לעזיבה: קיים סיכוי של 10% שלקוחות השוהים במערכת שעתיים או יותר יחליטו לעזוב</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Simulation Goals section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">מטרות הסימולציה</h3>
            <ul class="custom-list">
                <li>📊 הבנת השפעת הפרמטרים השונים על חוויית הלקוח</li>
                <li>🔍 חקירת תרחישים שונים וניתוח ביצועים</li>
                <li>🧠 פיתוח הבנה עמוקה של מערכות מורכבות</li>
                <li>📈 שיפור יעילות התהליכים</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Quote section
    st.markdown("""
        <div class="info-box rtl-content">
            <p style="font-style: italic;">"הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו."</p>
            <p style="text-align: right; color: #666;">- לא ידוע</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()