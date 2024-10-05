import streamlit as st
import numpy as np
from utils import set_ltr_sliders, set_rtl

def mid_square_method(seed, n):
    """Generate random numbers using Mid-Square method."""
    results = []
    for i in range(n):
        z_squared = str(seed ** 2).zfill(8)  # Square and pad to ensure 8 digits
        next_seed = int(z_squared[2:6])  # Take the middle 4 digits
        u = next_seed / 10000  # Generate U_i
        results.append((seed, u))
        seed = next_seed
    return results

def lcg_method(a, c, m, seed, n):
    """Generate random numbers using Linear Congruential Generator."""
    results = []
    for i in range(n):
        next_seed = (a * seed + c) % m
        u = next_seed / m  # Generate U_i
        results.append((seed, u))
        seed = next_seed
    return results


def show_random_generator():
    set_ltr_sliders() 
    # Streamlit UI
    st.title("Random Number Generators")

    # Mid-Square Method
    st.subheader("Mid-Square Method")
    seed_ms = st.slider("Enter a 4-digit seed for Mid-Square Method:", min_value=1000, max_value=9999, value=1234)
    n_ms = st.slider("How many numbers to generate?", min_value=100, max_value=10000, value=1000)

    if st.button("Generate Mid-Square Numbers"):
        mid_square_results = mid_square_method(seed_ms, n_ms)
        st.write("Mid-Square Method Results:")
        for i, (zi, ui) in enumerate(mid_square_results):
            st.write(f"Z_{i} = {zi}, U_{i} = {ui:.4f}")

    # Linear Congruential Generator (LCG)
    st.subheader("Linear Congruential Generator (LCG)")
    a = st.slider("Enter multiplier (a):", min_value=1, value=5)
    c = st.slider("Enter increment (c):", min_value=0, value=3)
    m = st.slider("Enter modulus (m):", min_value=2, value=16)
    seed_lcg = st.slider("Enter the seed (Z0):", min_value=0, value=7)
    n_lcg = st.slider("How many numbers to generate (LCG)?", min_value=1, max_value=100, value=10)

    if st.button("Generate LCG Numbers"):
        lcg_results = lcg_method(a, c, m, seed_lcg, n_lcg)
        st.write("LCG Results:")
        for i, (zi, ui) in enumerate(lcg_results):
            st.write(f"Z_{i} = {zi}, U_{i} = {ui:.4f}")


if __name__ == "__main__":
    show_random_generator()