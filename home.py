# story.py
import streamlit as st
import matplotlib.pyplot as plt
from styles import get_custom_css

def show():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>🚚 משאית האוכל העמוסה</h1>
            <p style="font-size: 1.2rem;">
                צאו למסע מרתק אל תוך פעילות היומיום של "משאית האוכל העמוסה".
                סימולציה אינטראקטיבית זו מבוססת על תורת התורים והתפלגויות הסתברותיות, המדמה את האתגרים של ניהול משאית אוכל רחוב במציאות.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">סקירת הסימולציה</h3>
            <p>צללו ליום טיפוסי ב"משאית האוכל העמוסה", תוך התמקדות באינטראקציות מגוונות עם לקוחות ודינמיקת הכנת המזון.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Elements Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">אלמנטים מרכזיים בסימולציה</h3>
            
            <h4>1. הגעת לקוחות 👥</h4>
            <ul class="custom-list">
                <li>הגעות אקראיות: לקוחות מגיעים באופן ספונטני, נמשכים לתפריט המפתה</li>
                <li>התפלגות מעריכית: מרווחי ההגעה ממוצעים כ-6 דקות, המכניסים אי-ודאות לתור</li>
            </ul>

            <h4>2. דינמיקת עמדת ההזמנות 📝</h4>
            <div class="metric-container">
                <p>לקוחות עם דחיפויות והעדפות שונות משפיעים על זמני עיבוד ההזמנות:</p>
                <table class="custom-table">
                    <tr>
                        <th>סוג לקוח</th>
                        <th>אחוז</th>
                        <th>זמן הזמנה</th>
                    </tr>
                    <tr>
                        <td>סוג A</td>
                        <td>50%</td>
                        <td>אחיד (3-4 דקות)</td>
                    </tr>
                    <tr>
                        <td>סוג B</td>
                        <td>25%</td>
                        <td>משולש (4-6 דקות)</td>
                    </tr>
                    <tr>
                        <td>סוג C</td>
                        <td>25%</td>
                        <td>קבוע (10 דקות)</td>
                    </tr>
                </table>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Meal Preparation Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h4>3. פרטי הכנת הארוחות 👨‍🍳</h4>
            <p><strong>בישול במנות:</strong></p>
            <div class="metric-container">
                <ul class="custom-list">
                    <li>השפים מכינים ארוחות במנות, המשפיעות על מהירות ואיכות השירות</li>
                    <li>זמני בישול סטוכסטיים:</li>
                    <ul>
                        <li>התפלגות נורמלית</li>
                        <li>ממוצע (μ): 5 דקות</li>
                        <li>סטיית תקן (σ): דקה אחת</li>
                    </ul>
                </ul>
            </div>

            <div class="info-box">
                <h4>הסתברויות לפי גודל המנה:</h4>
                <table class="custom-table">
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
    
    # Pickup and Customer Patience Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h4>4. זמן איסוף 🕒</h4>
            <p>התפלגות אחידה: זמני האיסוף נעים בין 2 ל-4 דקות</p>

            <h4>5. סבלנות הלקוחות 😊</h4>
            <div class="info-box">
                <p>הסתברות לעזיבה: קיים סיכוי של 10% שלקוחות השוהים במערכת שעתיים או יותר יחליטו לעזוב</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Conclusion Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">התנסות וחקירה</h3>
            <p>חוו את האתגרים התפעוליים והתרגשות ניהול משאית אוכל רחוב דרך הסימולציה האינטראקטיבית הזו. 
               מוכנים לנווט בהמולה והשגשוג? בואו נתחיל את החוויה ב"משאית האוכל העמוסה"! 🚚🍔🌭🍟</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()