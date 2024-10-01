import simpy
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from food_truck import run_simulation_with_speed  # Assuming this is the step-by-step simulation function
from logger import EventLogger  # Assuming EventLogger handles logging of events
from utils import set_rtl  # RTL setting function

# Call the set_rtl function to apply RTL styles
set_rtl()

def create_real_time_animation(df, current_step, max_queue_size):
    """
    Create a real-time bar chart showing the current state of the queues.
    """
    current_row = df.iloc[current_step]
    fig = go.Figure(data=[
        go.Bar(x=['Order Queue', 'Prep Queue', 'Pickup Queue', 'Total Queue'], 
               y=[current_row['Order Queue'], current_row['Prep Queue'], current_row['Pickup Queue'], current_row['Total Queue']],
               marker=dict(color=['blue', 'green', 'red', 'black']))
    ])
    
    fig.update_layout(
        title=f"Real-time Simulation at Step {current_step}",
        xaxis_title="Queue Type",
        yaxis_title="Queue Size",
        yaxis=dict(range=[0, max_queue_size])
    )
    
    return fig


def create_final_graphs(food_truck):
    """
    Create the final static graphs showing queue sizes over time.
    """
    df = pd.DataFrame({
        'Time': range(len(food_truck.queue_sizes['order'])),
        'Order Queue': food_truck.queue_sizes['order'],
        'Prep Queue': food_truck.queue_sizes['prep'],
        'Pickup Queue': food_truck.queue_sizes['pickup'],
        'Total Queue': food_truck.queue_sizes['total']
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Order Queue'], mode='lines', name='Order Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Prep Queue'], mode='lines', name='Prep Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Pickup Queue'], mode='lines', name='Pickup Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Total Queue'], mode='lines', name='Total Queue', line=dict(color='black', width=4)))

    fig.update_layout(title="Queue Sizes Over Time", xaxis_title="Time", yaxis_title="Queue Size", legend_title="Queue Type")
    
    return fig, df  # Return both the plot and the DataFrame


def show_food_truck():
    st.title("סימולציית משאית מזון")

    st.header("הגדרות סימולציה")
    sim_time = st.slider("זמן סימולציה (דקות)", 1000, 10000, 5000)
    arrival_rate = st.slider("זמן ממוצע בין הגעות לקוחות (דקות)", 5, 20, 10)
    order_time_min = st.slider("זמן הזמנה מינימלי (דקות)", 1, 5, 3)
    order_time_max = st.slider("זמן הזמנה מקסימלי (דקות)", 5, 10, 7)
    leave_probability = st.slider("הסתברות לעזיבה לפני הזמנה", 0.0, 0.5, 0.1)
    
    config = {
        'order_capacity': st.slider("כמות עמדות בהזמנה", 1, 5, 1),
        'prep_capacity': st.slider("כמות עמדות בהכנה", 1, 5, 1),
        'pickup_capacity': st.slider("כמות עמדות באיסוף", 1, 5, 1)
    }

    if st.button("הפעל סימולציה"):
        with st.spinner("מריץ סימולציה בזמן אמת..."):
            logger = EventLogger()
            speed = st.slider("מהירות סימולציה (פי)", 0.1, 10.0, 1.0)

            # Run the simulation in steps, showing real-time updates
            env = simpy.Environment()
            food_truck = run_simulation_with_speed(env, sim_time, arrival_rate, order_time_min, order_time_max, leave_probability, config, logger, speed)

            # Store simulation in session
            st.session_state.food_truck = food_truck

            # Retrieve queue data
            queue_data = pd.DataFrame({
                'Time': range(len(food_truck.queue_sizes['order'])),
                'Order Queue': food_truck.queue_sizes['order'],
                'Prep Queue': food_truck.queue_sizes['prep'],
                'Pickup Queue': food_truck.queue_sizes['pickup'],
                'Total Queue': food_truck.queue_sizes['total']
            })

            # Maximum queue size for the y-axis
            max_queue_size = queue_data[['Order Queue', 'Prep Queue', 'Pickup Queue', 'Total Queue']].max().max()

            # Real-time simulation loop
            st.subheader("תור בזמן אמת")
            for step in range(len(queue_data)):
                real_time_chart = create_real_time_animation(queue_data, step, max_queue_size)
                st.plotly_chart(real_time_chart, use_container_width=True)
                st.sleep(1.0 / speed)  # Control the speed of the animation
            
            st.success("הסימולציה בזמן אמת הושלמה!")

        # After real-time simulation, show the final graphs
        st.subheader("תוצאות הסימולציה (גרפים סופיים)")
        final_chart, final_data = create_final_graphs(food_truck)
        st.plotly_chart(final_chart, use_container_width=True)

    # Description at the end
    st.write("""
    #### חקרו את הסימולציה
    נסו וראו כיצד משתנים שונים משפיעים על ביצועי משאית המזון. התאימו את ההגדרות וההרצה לסימולציות שונות!
    """)


if __name__ == "__main__":
    show_food_truck()
