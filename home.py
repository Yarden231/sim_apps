# home.py
import streamlit as st
import matplotlib.pyplot as plt
from styles import get_custom_css

def show():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header with subtitle
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>🚚 משאית האוכל העמוסה</h1>
            <p>סימולציה אינטראקטיבית המבוססת על תורת התורים והתפלגויות הסתברותיות, המדמה את האתגרים של ניהול משאית אוכל רחוב</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Welcome section
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
        # Image container
        st.markdown('<div class="img-container rtl-content">', unsafe_allow_html=True)
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
            <div class="metric-container">
                <ul class="custom-list">
                    <li>הגעות אקראיות: לקוחות מגיעים באופן ספונטני, נמשכים לתפריט המפתה</li>
                    <li>התפלגות מעריכית: מרווחי ההגעה ממוצעים כ-6 דקות, המכניסים אי-ודאות לתור</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Order Station section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">2. דינמיקת עמדת ההזמנות 📝</h3>
            <p>לקוחות מגיעים עם דחיפויות והעדפות שונות המשפיעים על זמני עיבוד ההזמנות:</p>
            
            <div class="metric-container">
                <table class="styled-table">
                    <tr>
                        <th>סוג לקוח</th>
                        <th>אחוז מהלקוחות</th>
                        <th>זמן הזמנה</th>
                    </tr>
                    <tr>
                        <td>סוג A</td>
                        <td>50%</td>
                        <td>אחיד (3-4 דקות) - המהיר ביותר</td>
                    </tr>
                    <tr>
                        <td>סוג B</td>
                        <td>25%</td>
                        <td>משולש (4-6 דקות) - בינוני</td>
                    </tr>
                    <tr>
                        <td>סוג C</td>
                        <td>25%</td>
                        <td>קבוע (10 דקות) - האיטי ביותר</td>
                    </tr>
                </table>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Meal Preparation section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">3. פרטי הכנת הארוחות 👨‍🍳</h3>
            <p>השפים במשאית האוכל מכינים ארוחות במנות, דבר המשפיע הן על המהירות והן על איכות השירות.</p>
            
            <div class="info-box">
                <h4>זמני בישול סטוכסטיים:</h4>
                <ul class="custom-list">
                    <li>התפלגות נורמלית</li>
                    <li>ממוצע (μ): 5 דקות - זמן ממוצע להכנת מנה</li>
                    <li>סטיית תקן (σ): דקה אחת - משקף את השונות בזמני הבישול</li>
                </ul>
            </div>

            <div class="metric-container">
                <h4>הסתברויות לפי גודל המנה:</h4>
                <table class="styled-table">
                    <tr>
                        <th>גודל מנה</th>
                        <th>הסתברות</th>
                        <th>השפעה</th>
                        <th>סיכוי לבישול חסר</th>
                    </tr>
                    <tr>
                        <td>מנה בודדת</td>
                        <td>20%</td>
                        <td>שירות מהיר ואישי</td>
                        <td>0%</td>
                    </tr>
                    <tr>
                        <td>זוג מנות</td>
                        <td>50%</td>
                        <td>איזון בין מהירות ואיכות</td>
                        <td>0%</td>
                    </tr>
                    <tr>
                        <td>שלוש מנות</td>
                        <td>30%</td>
                        <td>יעילות גבוהה</td>
                        <td>30%</td>
                    </tr>
                </table>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Pickup and Customer Patience section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">4. זמן איסוף וסבלנות לקוחות ⏱️</h3>
            
            <div class="metric-container">
                <h4>זמן איסוף:</h4>
                <p>התפלגות אחידה: זמני האיסוף נעים בין 2 ל-4 דקות</p>
            </div>

            <div class="info-box">
                <h4>5. סבלנות הלקוחות 😊</h4>
                <p>הסתברות לעזיבה: קיים סיכוי של 10% שלקוחות השוהים במערכת שעתיים או יותר יחליטו לעזוב</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Simulation Goals section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">מטרות הסימולציה</h3>
            <div class="metric-container">
                <ul class="custom-list">
                    <li>📊 הבנת השפעת הפרמטרים השונים על חוויית הלקוח</li>
                    <li>🔍 חקירת תרחישים שונים וניתוח ביצועים</li>
                    <li>🧠 פיתוח הבנה עמוקה של מערכות מורכבות בתנאי אי-וודאות</li>
                    <li>📈 שיפור יעילות התהליכים במשאית</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quote box at the bottom
    st.markdown("""
        <div class="info-box rtl-content">
            <p style="font-style: italic;">"הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו."</p>
            <p style="text-align: right; color: #666;">- לא ידוע</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()