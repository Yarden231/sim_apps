import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def sample_uniform(a, b):
    u = random.random()
    return a + (b - a) * u

def sample_bernoulli(p):
    u = random.random()
    return 1 if u < p else 0

def sample_exponential(lambda_param):
    u = random.random()
    return -math.log(1 - u) / lambda_param

def sample_fair_die():
    u = random.random()
    if u < 1/6:
        return 1
    elif u < 2/6:
        return 2
    elif u < 3/6:
        return 3
    elif u < 4/6:
        return 4
    elif u < 5/6:
        return 5
    else:
        return 6

def sample_normal(mu, sigma):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z

def sample_composite_distribution():
    u = random.random()
    if u < 0.2:
        return sample_normal(0, 1)
    else:
        return sample_normal(3, 1)

def f(x):
    return 3 * x**2

def sample_acceptance_rejection():
    while True:
        x = random.random()
        y = random.random() * 3
        if y <= f(x):
            return x

def show_sampling_methods():
    st.title("שיטות דגימה מוסברות עם מימושים בפייתון")
    
    st.header("1. דגימה מהתפלגויות ידועות")
    
    st.subheader("התפלגות אחידה")
    a = st.slider("ערך מינימלי (a)", 0.0, 10.0, 0.0)
    b = st.slider("ערך מקסימלי (b)", a + 0.1, 20.0, 10.0)
    if st.button("דגום מהתפלגות אחידה"):
        result = sample_uniform(a, b)
        st.write(f"תוצאת הדגימה: {result:.4f}")
    
    st.subheader("התפלגות ברנולי")
    p = st.slider("הסתברות להצלחה (p)", 0.0, 1.0, 0.5)
    if st.button("דגום מהתפלגות ברנולי"):
        result = sample_bernoulli(p)
        st.write(f"תוצאת הדגימה: {result}")
    
    st.header("2. שיטת הטרנספורם ההופכי - מקרה רציף")
    lambda_param = st.slider("פרמטר למבדא להתפלגות מעריכית", 0.1, 5.0, 1.0)
    if st.button("דגום מהתפלגות מעריכית"):
        result = sample_exponential(lambda_param)
        st.write(f"תוצאת הדגימה: {result:.4f}")
    
    st.header("3. שיטת הטרנספורם ההופכי - מקרה בדיד")
    if st.button("הטל קובייה הוגנת"):
        result = sample_fair_die()
        st.write(f"תוצאת ההטלה: {result}")
    
    st.header("4. שיטת הקומפוזיציה")
    if st.button("דגום מהתפלגות מורכבת"):
        result = sample_composite_distribution()
        st.write(f"תוצאת הדגימה: {result:.4f}")
    
    st.header("5. שיטת הקבלה-דחייה")
    if st.button("דגום משיטת הקבלה-דחייה"):
        result = sample_acceptance_rejection()
        st.write(f"תוצאת הדגימה: {result:.4f}")

    st.header("השוואה ויזואלית")
    num_samples = st.slider("מספר דגימות", 100, 10000, 1000)
    
    methods = {
        "Uniform": lambda: sample_uniform(a, b),
        "Exponential": lambda: sample_exponential(lambda_param),
        "Composite": sample_composite_distribution,
        "Acceptance-Rejection": sample_acceptance_rejection
    }
    
    selected_methods = st.multiselect("בחר שיטות להשוואה", list(methods.keys()))
    
    if st.button("צור היסטוגרמות"):
        fig, axs = plt.subplots(len(selected_methods), 1, figsize=(10, 5*len(selected_methods)))
        if len(selected_methods) == 1:
            axs = [axs]
        
        for i, method in enumerate(selected_methods):
            samples = [methods[method]() for _ in range(num_samples)]
            axs[i].hist(samples, bins=50, density=True)
            axs[i].set_title(f"Histogram of {method} Distribution")
            axs[i].set_xlabel("Value")
            axs[i].set_ylabel("Density")
        
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    show_sampling_methods()
