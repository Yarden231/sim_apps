import streamlit as st
import simpy
import random
import plotly.graph_objs as go
import time
import pandas as pd
from utils import set_rtl, set_ltr_sliders

# Simulation Classes and Functions
class Employee:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.busy = False
        self.support_times = []

    def handle_call(self, call_duration):
        self.busy = True
        yield self.env.timeout(call_duration)
        self.busy = False
        self.support_times.append(call_duration)

class CallCenter:
    def __init__(self, env, num_employees):
        self.env = env
        self.employees = [Employee(env, i) for i in range(num_employees)]
        self.queue = simpy.Resource(env, capacity=num_employees)
        self.wait_times = []
        self.queue_lengths = []
        self.employee_utilization = []

    def request_employee(self, customer):
        with self.queue.request() as request:
            arrival_time = self.env.now
            yield request
            wait_time = self.env.now - arrival_time
            self.wait_times.append(wait_time)
            employee = self.get_free_employee()
            if employee:
                yield self.env.process(employee.handle_call(customer.call_duration))

    def get_free_employee(self):
        for employee in self.employees:
            if not employee.busy:
                return employee
        return None

    def track_metrics(self):
        while True:
            self.queue_lengths.append(len(self.queue.queue))
            utilization = sum([emp.busy for emp in self.employees]) / len(self.employees)
            self.employee_utilization.append(utilization)
            yield self.env.timeout(1)

class Customer:
    def __init__(self, env, call_center, call_duration):
        self.env = env
        self.call_center = call_center
        self.call_duration = call_duration

    def request_support(self):
        yield self.env.process(self.call_center.request_employee(self))

def generate_customers(env, call_center, interval, call_duration_mean):
    while True:
        yield env.timeout(random.expovariate(1.0 / interval))
        call_duration = random.expovariate(1.0 / call_duration_mean)
        customer = Customer(env, call_center, call_duration)
        env.process(customer.request_support())

# Plot real-time queue updates
def plot_real_time_queues(call_center, step):
    # Create DataFrame for real-time data
    df = pd.DataFrame({
        'Time': range(len(call_center.queue_lengths)),
        'Queue Length': call_center.queue_lengths,
        'Employee Utilization': [u * 100 for u in call_center.employee_utilization]
    })

    # Current metrics for dynamic title
    current_queue_size = df['Queue Length'].iloc[step]
    current_utilization = df['Employee Utilization'].iloc[step]

    # Create bar chart to visualize real-time queue length
    fig = go.Figure(data=[
        go.Bar(x=['Queue Length', 'Employee Utilization (%)'], 
               y=[current_queue_size, current_utilization],
               marker=dict(color=['blue', 'green']))
    ])

    # Update layout with dynamic title
    fig.update_layout(
        title=f"Queue & Utilization Status at Step {step}: Queue={current_queue_size}, Utilization={current_utilization:.2f}%",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 100])  # Set y-axis limit
    )
    return fig

# Plot after the simulation finishes
def plot_final_metrics(call_center):
    df = pd.DataFrame({
        'Time': range(len(call_center.queue_lengths)),
        'Queue Length': call_center.queue_lengths,
        'Employee Utilization (%)': [u * 100 for u in call_center.employee_utilization]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'], 
        y=df['Queue Length'], 
        mode='lines', 
        name='Queue Length', 
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'], 
        y=df['Employee Utilization (%)'], 
        mode='lines', 
        name='Employee Utilization (%)', 
        line=dict(color='green')
    ))

    # Update layout for final plot
    fig.update_layout(
        title="Queue Length and Employee Utilization Over Time",
        xaxis_title="Time (Steps)",
        yaxis_title="Value",
        legend_title="Metrics"
    )
    return fig

# Main simulation function with real-time updates
def run_simulation(num_employees, customer_interval, call_duration_mean, simulation_time, real_time_chart, progress_placeholder):
    env = simpy.Environment()
    call_center = CallCenter(env, num_employees)
    env.process(generate_customers(env, call_center, customer_interval, call_duration_mean))
    env.process(call_center.track_metrics())

    for step in range(simulation_time):
        env.run(until=step + 1)  # Step simulation by one unit of time

        # Update real-time plot
        if step < len(call_center.queue_lengths):  # Ensure there is enough data
            chart = plot_real_time_queues(call_center, step)
            real_time_chart.plotly_chart(chart, use_container_width=True)

        # Update progress bar
        progress_placeholder.progress((step + 1) / simulation_time)
        time.sleep(0.1)  # Control speed of real-time updates

    return call_center

# Streamlit app function
def show():
    # Set RTL for Hebrew text layout (if needed)
    set_rtl()
    set_ltr_sliders()  # Ensure sliders are LTR

    st.title("סימולציית מרכז שירות לקוחות בזמן אמת")

    # Simulation parameters in RTL
    st.header("הגדרות סימולציה")
    num_employees = st.slider("מספר נציגי שירות", 1, 10, 3)
    customer_interval = st.slider("זמן ממוצע בין הגעות לקוחות (בדקות)", 1, 10, 4)
    call_duration_mean = st.slider("משך שיחה ממוצע (בדקות)", 1, 10, 8)
    simulation_time = st.slider("זמן סימולציה (ביחידות)", 100, 1000, 400)

    if st.button("הפעל סימולציה"):
        # Create placeholders for progress and real-time chart
        progress_placeholder = st.empty()
        real_time_chart = st.empty()

        st.write("מריץ סימולציה בזמן אמת...")

        # Run the simulation
        call_center = run_simulation(num_employees, customer_interval, call_duration_mean, simulation_time, real_time_chart, progress_placeholder)

        st.success("הסימולציה הושלמה!")

        # Final plot after simulation
        st.header("גודל התור וניצולת עובדים לאורך זמן")
        final_chart = plot_final_metrics(call_center)
        st.plotly_chart(final_chart, use_container_width=True)

# Ensure the app runs directly
if __name__ == "__main__":
    show()
