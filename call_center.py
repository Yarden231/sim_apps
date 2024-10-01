import streamlit as st
import simpy
import random
import statistics
import plotly.graph_objs as go
import time
from utils import set_rtl  
from utils import set_ltr_sliders

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

# Real-time plot for live queue updates (Queue Length only)
def plot_real_time_queues(call_center, step):
    queue_size = call_center.queue_lengths[step] if step < len(call_center.queue_lengths) else 0

    fig = go.Figure(data=[
        go.Bar(x=['Queue Size'], y=[queue_size], marker=dict(color='blue'))
    ])

    fig.update_layout(
        title=f"Queue Length at Step {step}",
        xaxis_title="Metric",
        yaxis_title="Queue Size",
        yaxis=dict(range=[0, max(call_center.queue_lengths + [10])])  # Dynamic range adjustment
    )
    return fig

# After simulation plot: queue sizes and utilization over time
def plot_queue_sizes_over_time(call_center, simulation_time):
    time_points = list(range(simulation_time))

    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(x=time_points, y=call_center.queue_lengths, mode='lines', name='Queue Length', line=dict(color='blue')))
    fig_queue.add_trace(go.Scatter(x=time_points, y=[u * 100 for u in call_center.employee_utilization], mode='lines', name='Employee Utilization (%)', line=dict(color='green')))
    
    fig_queue.update_layout(
        title="Queue Length and Employee Utilization Over Time",
        xaxis_title="Time (Steps)",
        yaxis_title="Value",
        legend_title="Metrics"
    )
    return fig_queue

# Main simulation function with real-time updates
def run_simulation(num_employees, customer_interval, call_duration_mean, simulation_time, real_time_chart, progress_placeholder):
    env = simpy.Environment()
    call_center = CallCenter(env, num_employees)
    env.process(generate_customers(env, call_center, customer_interval, call_duration_mean))
    env.process(call_center.track_metrics())

    for step in range(simulation_time):
        env.step()  # Step simulation

        # Update real-time plot
        chart = plot_real_time_queues(call_center, step)
        real_time_chart.plotly_chart(chart, use_container_width=True)

        # Update progress bar
        progress_placeholder.progress((step + 1) / simulation_time)
        time.sleep(0.1)

    return call_center

# Streamlit app function
def show():
    # Set RTL for Hebrew text layout
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
        # Create placeholders for progress and real-time charts
        progress_placeholder = st.empty()
        real_time_chart = st.empty()

        st.write("מריץ סימולציה בזמן אמת...")

        # Run the simulation
        call_center = run_simulation(num_employees, customer_interval, call_duration_mean, simulation_time, real_time_chart, progress_placeholder)

        st.success("הסימולציה הושלמה!")

        # Final plot after simulation
        st.header("גודל תור וניצולת עובדים לאורך זמן")
        final_chart = plot_queue_sizes_over_time(call_center, simulation_time)
        st.plotly_chart(final_chart, use_container_width=True)

# Ensure the app runs directly
if __name__ == "__main__":
    show()
