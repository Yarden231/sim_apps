import streamlit as st

def show():
    st.header("Simulation Concepts Quiz")
    q1 = st.radio(
        "What is Monte Carlo simulation?",
        ["A type of probability distribution", "A statistical sampling technique", "A deterministic modeling approach"]
    )
    # Add more quiz questions and logic for scoring
