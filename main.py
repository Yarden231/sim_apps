import streamlit as st

# Set page config at the very top of the script
st.set_page_config(
    page_title="פלטפורמת קורס סימולציה",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pages import show_home, show_theory, show_quiz, show_project_guidelines
from simulations import show_call_center, show_food_truck
#from labs.lab_2 import show_theory_lab
from sampling_methods import show_sampling_methods

def main():
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
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'דף הבית'

    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

    pages = {
        "דף הבית": show_home,
        "תיאוריה": show_theory,
        #"מעבדת תיאוריה": show_theory_lab,
        #"אפליקציה אינטראקטיבית": show_interactive_simulation,
        "סימולציית מוקד שירות": show_call_center,
        "סימולציית משאית מזון": show_food_truck,
        "שיטות דגימה": show_sampling_methods,
        "בוחן": show_quiz,
        "הנחיות פרויקט": show_project_guidelines
    }

    selected_page = st.sidebar.radio("ניווט", list(pages.keys()), index=list(pages.keys()).index(st.session_state.page))

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Update session state with selected page
    st.session_state.page = selected_page

    # Display the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()