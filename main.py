import streamlit as st



# Set page config without the theme argument
st.set_page_config(
    page_title="Simulation Course Platform",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


from utils import set_rtl
# Call the set_rtl function to apply RTL styles
set_rtl()
# Import all page functions
from home import show as show_home
from theory import show as show_theory
from call_center import show as show_call_center
from food_truck2 import show_food_truck
from logger import EventLogger
from sampling_methods import show_sampling_methods


def main():
    # Hide default menu
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    # Sidebar and content styling
    st.markdown(
        """
        <style>
        body {
            background-color: #654321;  /* צבע רקע של הדף */
            color: #FFFFFF;  /* צבע טקסט של הדף */
        }

        .sidebar .sidebar-content {
            background-color: #8B4513;  /* צבע רקע של הסיידבר */
            color: #FFFFFF;  /* צבע טקסט בסיידבר */
        }

        .stButton>button {
            width: 100%;
            background-color: #D2B48C;  /* צבע רקע של הכפתור */
            border: none;
            color: #654321;  /* צבע טקסט של הכפתור */
            padding: 12px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 0;
            cursor: pointer;
            border-radius: 8px;
            transition-duration: 0.3s;
        }

        .stButton>button:hover {
            background-color: #C1A378;  /* צבע רקע של הכפתור במעבר עכבר */
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FFFFFF;  /* צבע טקסט של כותרות */
        }

        .stMarkdown p {
            color: #FFFFFF;  /* צבע טקסט של פסקאות */
            font-size: 16px;
            line-height: 1.6;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'דף הבית'

    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

    # Define the available pages and their corresponding functions
    pages = {
        "דף הבית": show_home,
        "תיאוריה": show_theory,
        "סימולציית מוקד שירות": show_call_center,
        "סימולציית משאית מזון": show_food_truck,
        "שיטות דגימה": show_sampling_methods,
    }

    # Add buttons for each page in the sidebar
    for page_name, page_func in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_name

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Display the selected page's content
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()
