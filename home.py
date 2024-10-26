# story.py
import streamlit as st
#from styles import get_custom_css

def get_custom_css():
    return """
    <style>
        /* Existing styles ... */
        
        /* RTL Support */
        .rtl-content {
            direction: rtl;
            text-align: right;
        }
        
        /* Table styles */
        .dataframe {
            direction: rtl;
            text-align: right;
        }
        
        .dataframe th {
            text-align: right;
            font-weight: bold;
        }
        
        /* Info box styling */
        .stInfo {
            direction: rtl;
            text-align: right;
        }
        
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            direction: rtl;
            text-align: right;
        }
    </style>
    """


def show():
    # Apply custom CSS
    st.write(get_custom_css(), unsafe_allow_html=True)
    
    # Main header using Streamlit components
    st.markdown('<div class="custom-header rtl-content">', unsafe_allow_html=True)
    st.title("🚚 משאית האוכל העמוסה")
    st.markdown("""
        <p style="font-size: 1.2rem; direction: rtl;">
            צאו למסע מרתק אל תוך פעילות היומיום של "משאית האוכל העמוסה".
            סימולציה אינטראקטיבית זו מבוססת על תורת התורים והתפלגויות הסתברותיות, המדמה את האתגרים של ניהול משאית אוכל רחוב במציאות.
        </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Simulation Overview
    st.markdown("### סקירת הסימולציה")
    st.markdown("צללו ליום טיפוסי ב\"משאית האוכל העמוסה\", תוך התמקדות באינטראקציות מגוונות עם לקוחות ודינמיקת הכנת המזון.")
    
    # Key Elements Section
    st.markdown("### אלמנטים מרכזיים בסימולציה")
    
    # Customer Arrivals
    st.markdown("#### 1. הגעת לקוחות 👥")
    st.markdown("""
    * הגעות אקראיות: לקוחות מגיעים באופן ספונטני, נמשכים לתפריט המפתה
    * התפלגות מעריכית: מרווחי ההגעה ממוצעים כ-6 דקות, המכניסים אי-ודאות לתור
    """)
    
    # Order Station Dynamics
    st.markdown("#### 2. דינמיקת עמדת ההזמנות 📝")
    st.markdown("לקוחות עם דחיפויות והעדפות שונות משפיעים על זמני עיבוד ההזמנות:")
    
    # Create a table using Streamlit
    order_data = {
        "סוג לקוח": ["סוג A", "סוג B", "סוג C"],
        "אחוז": ["50%", "25%", "25%"],
        "זמן הזמנה": ["אחיד (3-4 דקות)", "משולש (4-6 דקות)", "קבוע (10 דקות)"]
    }
    st.table(order_data)
    
    # Meal Preparation Section
    st.markdown("#### 3. פרטי הכנת הארוחות 👨‍🍳")
    st.markdown("**בישול במנות:**")
    st.markdown("""
    * השפים מכינים ארוחות במנות, המשפיעות על מהירות ואיכות השירות
    * זמני בישול סטוכסטיים:
        * התפלגות נורמלית
        * ממוצע (μ): 5 דקות
        * סטיית תקן (σ): דקה אחת
    """)
    
    # Batch probabilities table
    st.markdown("**הסתברויות לפי גודל המנה:**")
    batch_data = {
        "גודל מנה": ["מנה בודדת", "זוג מנות", "שלוש מנות"],
        "הסתברות": ["20%", "50%", "30%"],
        "השפעה": ["שירות מהיר ואישי", "איזון בין מהירות ואיכות", "יעילות גבוהה"],
        "סיכוי לבישול חסר": ["0%", "0%", "30%"]
    }
    st.table(batch_data)
    
    # Pickup Time Section
    st.markdown("#### 4. זמן איסוף 🕒")
    st.markdown("התפלגות אחידה: זמני האיסוף נעים בין 2 ל-4 דקות")
    
    # Customer Patience Section
    st.markdown("#### 5. סבלנות הלקוחות 😊")
    st.info("הסתברות לעזיבה: קיים סיכוי של 10% שלקוחות השוהים במערכת שעתיים או יותר יחליטו לעזוב")
    
    # Conclusion Section
    st.markdown("### התנסות וחקירה")
    st.markdown("""
    חוו את האתגרים התפעוליים והתרגשות ניהול משאית אוכל רחוב דרך הסימולציה האינטראקטיבית הזו.
    מוכנים לנווט בהמולה והשגשוג? בואו נתחיל את החוויה ב"משאית האוכל העמוסה"! 🚚🍔🌭🍟
    """)

if __name__ == "__main__":
    show()