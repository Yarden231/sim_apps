# home.py
import streamlit as st
import matplotlib.pyplot as plt
from styles import get_custom_css

def set_rtl():
    st.markdown("""
        <style>
            .element-container, .stMarkdown, .stText {
                direction: rtl;
                text-align: right;
            }
            .stSelectbox > div > div > div {
                direction: rtl;
                text-align: right;
            }
        </style>
    """, unsafe_allow_html=True)

def show():
    set_rtl()
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="custom-header" style="text-align: right; direction: rtl;">
            <h1 style="direction: rtl;">🚚 משאית האוכל העמוסה</h1>
            <p style="direction: rtl;">סימולציה אינטראקטיבית המבוססת על תורת התורים והתפלגויות הסתברותיות, המדמה את האתגרים של ניהול משאית אוכל רחוב</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div class="custom-card" style="direction: rtl; text-align: right;">
                <h3 class="section-header">ברוכים הבאים!</h3>
                <p>
                    צאו למסע מרתק אל תוך פעילות היומיום של משאית האוכל העמוסה. 
                    בדוגמה זו נחקור את האתגרים היומיומיים של משאית אוכל מבוקשת, 
                    בה מתחלפים לקוחות, נרשמות הזמנות ומתבצעת הכנה מרובת שלבים.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box" style="direction: rtl; text-align: right;">
                <h4>מבנה המערכת:</h4>
                <ul class="custom-list" style="padding-right: 20px; margin-right: 0;">
                    <li>🎯 עמדת הזמנה</li>
                    <li>👨‍🍳 עמדת הכנת מנות</li>
                    <li>📦 עמדת אריזה והגשה</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        try:
            image = plt.imread("food_track_image.jpg")
            st.image(image, caption="משאית המזון שלנו", use_column_width=True)
        except:
            st.info("תמונת משאית המזון לא נמצאה")
        st.markdown('</div>', unsafe_allow_html=True)

    # Customer Arrival section
    st.markdown("""
        <div class="custom-card" style="direction: rtl; text-align: right;">
            <h3 class="section-header">1. הגעת לקוחות 👥</h3>
            <div class="metric-container" style="text-align: right;">
                <ul class="custom-list" style="padding-right: 20px; margin-right: 0;">
                    <li>הגעות אקראיות: לקוחות מגיעים באופן ספונטני</li>
                    <li>התפלגות מעריכית: מרווחי הגעה ממוצעים של 6 דקות</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Order Types section
    st.markdown("""
        <div class="custom-card" style="direction: rtl; text-align: right;">
            <h3 class="section-header">2. סוגי הזמנות 📝</h3>
            <div class="metric-container" style="text-align: right;">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; direction: rtl;">
                    <div style="font-weight: bold;">סוג לקוח</div>
                    <div style="font-weight: bold;">אחוז</div>
                    <div style="font-weight: bold;">זמן הזמנה</div>
                    
                    <div>סוג A</div>
                    <div>50%</div>
                    <div>אחיד (3-4 דקות)</div>
                    
                    <div>סוג B</div>
                    <div>25%</div>
                    <div>משולש (4-6 דקות)</div>
                    
                    <div>סוג C</div>
                    <div>25%</div>
                    <div>קבוע (10 דקות)</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Meal Preparation section
    st.markdown("""
        <div class="custom-card" style="direction: rtl; text-align: right;">
            <h3 class="section-header">3. הכנת ארוחות 👨‍🍳</h3>
            
            <div class="info-box" style="direction: rtl; text-align: right;">
                <h4>זמני בישול:</h4>
                <ul class="custom-list" style="padding-right: 20px; margin-right: 0;">
                    <li>התפלגות נורמלית</li>
                    <li>ממוצע: 5 דקות</li>
                    <li>סטיית תקן: דקה אחת</li>
                </ul>
            </div>

            <div class="metric-container" style="text-align: right;">
                <h4>הסתברויות לפי גודל מנה:</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; direction: rtl;">
                    <div style="font-weight: bold;">גודל מנה</div>
                    <div style="font-weight: bold;">הסתברות</div>
                    <div style="font-weight: bold;">השפעה</div>
                    <div style="font-weight: bold;">סיכוי לבישול חסר</div>
                    
                    <div>מנה בודדת</div>
                    <div>20%</div>
                    <div>שירות מהיר</div>
                    <div>0%</div>
                    
                    <div>זוג מנות</div>
                    <div>50%</div>
                    <div>מאוזן</div>
                    <div>0%</div>
                    
                    <div>שלוש מנות</div>
                    <div>30%</div>
                    <div>יעילות גבוהה</div>
                    <div>30%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Pickup and Patience section
    st.markdown("""
        <div class="custom-card" style="direction: rtl; text-align: right;">
            <h3 class="section-header">4. איסוף וסבלנות לקוחות ⏱️</h3>
            
            <div class="metric-container" style="text-align: right;">
                <h4>זמן איסוף:</h4>
                <p>התפלגות אחידה: 2-4 דקות</p>
            </div>

            <div class="info-box" style="direction: rtl; text-align: right;">
                <h4>סבלנות לקוחות:</h4>
                <p>10% סיכוי לעזיבה אחרי שעתיים</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Goals section
    st.markdown("""
        <div class="custom-card" style="direction: rtl; text-align: right;">
            <h3 class="section-header">מטרות הסימולציה</h3>
            <ul class="custom-list" style="padding-right: 20px; margin-right: 0;">
                <li>📊 הבנת השפעת פרמטרים על חוויית לקוח</li>
                <li>🔍 חקירת תרחישים שונים</li>
                <li>🧠 הבנת מערכות מורכבות</li>
                <li>📈 שיפור יעילות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Quote section
    st.markdown("""
        <div class="info-box" style="direction: rtl; text-align: right;">
            <p style="font-style: italic; text-align: center;">"הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו."</p>
            <p style="text-align: left; color: #666;">- לא ידוע</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()