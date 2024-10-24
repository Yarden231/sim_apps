# home.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

def apply_custom_css():
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        
        /* Header styling */
        .title-container {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white !important;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Section headers */
        .section-header {
            color: #2c3e50 !important;
            border-bottom: 2px solid #ff6b6b;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Quote styling */
        .quote-container {
            background-color: #f1f8ff;
            border-left: 4px solid #4a90e2;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 10px 10px 0;
        }
        
        /* List styling */
        .custom-list {
            margin-right: 1.5rem;
        }
        
        /* Text direction for Hebrew */
        .hebrew-text {
            direction: rtl;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

def show():
    apply_custom_css()
    
    # Main title section
    st.markdown("""
        <div class="title-container">
            <h1>The Busy Food Truck 🚚</h1>
            <p style="font-size: 1.2rem;">סימולציה מתקדמת לניהול משאית מזון</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Introduction section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div class="card hebrew-text">
                <h3 class="section-header">ברוכים הבאים!</h3>
                <p># מערכת השירות של משאית המזון כוללת שלושה עמדות שירות:</p>
                <ul class="custom-list">
                    <li>🎯 עמדת הזמנה</li>
                    <li>👨‍🍳 עמדת הכנת מנות טעימות</li>
                    <li>📦 עמדת אריזה והגשה ללקוחות הנלהבים</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            image = plt.imread("food_track_image.jpg")
            st.image(image, use_column_width=True, caption="משאית המזון שלנו")
        except:
            st.info("תמונת משאית המזון לא נמצאה. אנא וודאו שהקובץ 'food_track_image.jpg' קיים בתיקייה.")
    
    # Simulation goals section
    st.markdown("""
        <div class="card hebrew-text">
            <h3 class="section-header">מטרות הסימולציה</h3>
            <ul class="custom-list">
                <li>📊 הבנת השפעת הפרמטרים השונים על חוויית הלקוח</li>
                <li>🔍 חקירת תרחישים שונים וניתוח ביצועים</li>
                <li>🧠 פיתוח הבנה עמוקה של מערכות מורכבות בתנאי אי-וודאות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Quote section
    st.markdown("""
        <div class="quote-container hebrew-text">
            <h4>ציטוט היום ✨</h4>
            <p style="font-style: italic;">"הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו."</p>
            <p style="text-align: left; color: #666;">- לא ידוע</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Course objectives section
    st.markdown("""
        <div class="card hebrew-text">
            <h3 class="section-header">מטרות הקורס</h3>
            <p>מטרת הקורס היא להקנות לסטודנטים את הכלים והידע הדרושים לבניית פרויקטי סימולציה בצורה פשוטה ומעשית.</p>
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>נושאי הקורס העיקריים:</h4>
                <ul class="custom-list">
                    <li>🔄 מידול המערכת</li>
                    <li>📥 יצירת קלט לסימולציה</li>
                    <li>🚀 הרצת ניסוי</li>
                    <li>📊 ניתוח פלט ותוצאות</li>
                </ul>
            </div>
            <p style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                הקורס משלב תאוריה ופרקטיקה עם יישום מעשי בשפת פייתון. בפשטות – הקורס מלמד איך לבצע סימולציה מא' ועד ת'.
            </p>
        </div>
    """, unsafe_allow_html=True)