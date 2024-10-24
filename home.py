# home.py
import streamlit as st
import matplotlib.pyplot as plt
from styles import get_custom_css

def show():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>🚚 The Busy Food Truck</h1>
            <p style="font-size: 1.2rem;">סימולציה מתקדמת לניהול משאית מזון</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">ברוכים הבאים!</h3>
                <p>מערכת השירות של משאית המזון כוללת שלושה עמדות שירות:</p>
                <ul class="custom-list">
                    <li>🎯 עמדת הזמנה</li>
                    <li>👨‍🍳 עמדת הכנת מנות טעימות</li>
                    <li>📦 עמדת אריזה והגשה ללקוחות הנלהבים</li>
                </ul>
            </div>
            
            <div class="info-box rtl-content">
                <p>דרך עמודי האפליקציה, תגלו כיצד באמצעות סימולציה ממחושבת, ניתן לדמות ולנתח תהליכים של זרימת לקוחות ולהבין כיצד ניהול תהליכי הזמנות, הכנת ארוחות ואיסוף משפיעים על הביצועים הכוללים של המערכת.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="img-container rtl-content">', unsafe_allow_html=True)
        try:
            image = plt.imread("food_track_image.jpg")
            st.image(image, use_column_width=True, caption="משאית המזון שלנו")
        except:
            st.info("תמונת משאית המזון לא נמצאה. אנא וודאו שהקובץ 'food_track_image.jpg' קיים בתיקייה.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Simulation goals section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">מטרות הסימולציה</h3>
            <div class="metric-container">
                <ul class="custom-list">
                    <li>📊 הבנת השפעת הפרמטרים השונים על חוויית הלקוח</li>
                    <li>🔍 חקירת תרחישים שונים וניתוח ביצועים</li>
                    <li>🧠 פיתוח הבנה עמוקה של מערכות מורכבות בתנאי אי-וודאות</li>
                </ul>
            </div>
        </div>
        
        <div class="custom-card rtl-content">
            <h3 class="section-header">מטרות הקורס</h3>
            <p>מטרת הקורס היא להקנות לסטודנטים את הכלים והידע הדרושים לבניית פרויקטי סימולציה בצורה פשוטה ומעשית.</p>
            <div class="metric-container">
                <h4>נושאי הקורס העיקריים:</h4>
                <ul class="custom-list">
                    <li>🔄 מידול המערכת</li>
                    <li>📥 יצירת קלט לסימולציה</li>
                    <li>🚀 הרצת ניסוי</li>
                    <li>📊 ניתוח פלט ותוצאות</li>
                </ul>
            </div>
        </div>
        
        <div class="info-box rtl-content">
            <p style="font-style: italic;">"הדרך הטובה ביותר לחזות את העתיד היא לסמלץ אותו."</p>
            <p style="text-align: right; color: #666;">- לא ידוע</p> <!-- Changed to right alignment -->
        </div>
    """, unsafe_allow_html=True)
