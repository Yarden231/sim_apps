import streamlit as st

# Set page config at the very top of the script
st.set_page_config(
    page_title="פלטפורמת קורס סימולציה",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import all page functions
from home import show as show_home
from theory import show as show_theory
from call_center import show as show_call_center
from food_truck import run_simulation, run_simulation_with_speed
from logger import EventLogger
from visualizations import  show_food_truck
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

    # Sidebar styling
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #2E4057;
            color: #E8EEF2;
        }
        .sidebar-nav {
            padding-top: 20px;
        }
        .stButton>button {
            width: 100%;
            background-color: #66A182;
            border: none;
            color: #E8EEF2;
            padding: 12px 20px;
            text-align: right;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 0;
            cursor: pointer;
            border-radius: 8px;
            transition-duration: 0.3s;
        }
        .stButton>button:hover {
            background-color: #7FB69E;
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
