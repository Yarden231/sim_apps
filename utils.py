# You can add utility functions here that might be used across different modules
# For example, data processing functions, custom chart creation functions, etc.
import streamlit as st

# Function to inject custom CSS for RTL layout
def set_rtl():
    st.markdown(
        """
        <style>
        body, .css-1lcbmhc, .css-1e5imcs, .css-1wif8ho { 
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def create_custom_chart(data):
    # Implement custom chart creation logic
    pass

def process_simulation_data(data):
    # Implement data processing logic
    pass
