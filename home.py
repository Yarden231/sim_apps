import streamlit as st
import matplotlib.pyplot as plt

def show():
    # Custom CSS to set text color to black for all elements
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header section
    st.header("ברוכים הבאים לפלטפורמת קורס סימולציה")

    # Introduction section
    st.write("""
    ##### פלטפורמה זו נועדה לסייע לכם ללמוד וליישם מושגי סימולציה באופן אפקטיבי ומעשי.
    השתמשו בסרגל הצד כדי לנווט בין החלקים השונים של הקורס ולגשת למשאבים מגוונים.
    """)

    # Course objectives section
    st.subheader("מטרות הקורס")
    st.write("""
            ###### מטרת הקורס היא להקנות לסטודנטים את הכלים והידע הדרושים לבניית פרויקטי סימולציה בצורה פשוטה ומעשית, תוך שימוש במחשב לדימוי וניתוח תהליכים מהעולם האמיתי.
            ###### הקורס מתמקד בנושאים הבאים:
            - **מידול המערכת**  
            - **יצירת קלט לסימולציה**  
            - **הרצת ניסוי**  
            - **ניתוח פלט ותוצאות**
            ###### הקורס משלב תאוריה ופרקטיקה עם יישום מעשי בשפת פייתון. בפשטות – הקורס מלמד איך לבצע סימולציה מא' ועד ת'.
            """)

    # Example course simulation section
    st.subheader("דוגמת הקורס: The Busy Food Truck")
    image= plt.imread("food_track_image.jpg")
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("""
    ###### האפליקציה תאפשר לכם לחקור את כל שלבי הסימולציה דרך דוגמה של מערכת שירות תוססת. תגלו כיצד זרימת לקוחות, ניהול הזמנות, הכנת ארוחות ואיסוף משפיעים על ביצועי המערכת.         
    #### מטרות הסימולציה:
    - הבנת השפעת הפרמטרים השונים (זמני המתנה, התפלגויות זמן שירות וכו') על חוויית הלקוח.
    - חקירת תרחישים שונים וניתוח ביצועים.
    - פיתוח הבנה עמוקה של מערכות מורכבות בתנאי אי-וודאות.
    """)

    # Motivational Quote section
    st.subheader("ציטוט היום")
    st.write("""
    ###### "הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו." - לא ידוע
    """)
