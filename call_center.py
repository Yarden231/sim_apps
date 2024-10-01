import streamlit as st
import simpy
import random
import statistics
import plotly.graph_objs as go
import time

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

def run_simulation(num_employees, customer_interval, call_duration_mean, simulation_time, progress_placeholder, chart_placeholder):
    env = simpy.Environment()
    call_center = CallCenter(env, num_employees)
    env.process(generate_customers(env, call_center, customer_interval, call_duration_mean))
    env.process(call_center.track_metrics())

    # Step-by-step simulation for real-time updates
    for i in range(simulation_time):
        env.step()  # Step simulation
        # Update real-time progress and chart
        update_real_time_chart(call_center, i, chart_placeholder)
        progress_placeholder.progress((i + 1) / simulation_time)
        time.sleep(0.1)  # Simulate real-time updates

    # Return the results after the simulation completes
    if call_center.wait_times:
        avg_wait = statistics.mean(call_center.wait_times)
    else:
        avg_wait = 0

    employee_utilization_percentages = [util * 100 for util in call_center.employee_utilization]

    return avg_wait, call_center.queue_lengths, employee_utilization_percentages

def update_real_time_chart(call_center, step, chart_placeholder):
    # Real-time queue length and employee utilization update
    current_queue_size = call_center.queue_lengths[step] if step < len(call_center.queue_lengths) else 0
    current_utilization = call_center.employee_utilization[step] * 100 if step < len(call_center.employee_utilization) else 0

    fig = go.Figure(data=[
        go.Bar(x=['Queue Length', 'Employee Utilization (%)'], 
               y=[current_queue_size, current_utilization], 
               marker=dict(color=['blue', 'green']))
    ])

    # Update the layout with dynamic title
    fig.update_layout(
        title=f"Real-Time Queue and Utilization at Step {step}",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, max(current_queue_size + 10, 100)])  # Dynamically scale the y-axis
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def plot_final_metrics(queue_lengths, employee_utilization, simulation_time):
    time_points = list(range(simulation_time))

    # Plot Queue Length Over Time
    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(x=time_points, y=queue_lengths, mode='lines', name='Queue Length', line=dict(color='blue')))
    fig_queue.update_layout(
        title="Queue Length Over Time",
        xaxis_title="Time",
        yaxis_title="Queue Length",
        legend_title="Metrics"
    )
    st.plotly_chart(fig_queue, use_container_width=True)

    # Plot Employee Utilization Over Time
    fig_utilization = go.Figure()
    fig_utilization.add_trace(go.Scatter(x=time_points, y=employee_utilization, mode='lines', name='Employee Utilization (%)', line=dict(color='green')))
    fig_utilization.update_layout(
        title="Employee Utilization Over Time",
        xaxis_title="Time",
        yaxis_title="Employee Utilization (%)",
        legend_title="Metrics"
    )
    st.plotly_chart(fig_utilization, use_container_width=True)

# Encapsulate Streamlit app in a function
def show():
    st.title("Real-Time Call Center Simulation")

    # Slider inputs for simulation parameters
    num_employees = st.slider("Number of Employees", min_value=1, max_value=10, value=3)
    customer_interval = st.slider("Average Time Between Customer Arrivals (Minutes)", min_value=1, max_value=10, value=4)
    call_duration_mean = st.slider("Average Call Duration (Minutes)", min_value=1, max_value=10, value=8)
    simulation_time = st.slider("Simulation Time (Steps)", min_value=100, max_value=1000, value=400)

    # Display the selected parameters
    st.write("### Selected Simulation Parameters:")
    st.write(f"- Number of Employees: {num_employees}")
    st.write(f"- Average Time Between Arrivals: {customer_interval} minutes")
    st.write(f"- Average Call Duration: {call_duration_mean} minutes")
    st.write(f"- Simulation Time: {simulation_time} steps")

    # Placeholder for progress bar and real-time chart
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Run simulation when button is pressed
    if st.button("Run Simulation"):
        avg_wait, queue_lengths, employee_utilization = run_simulation(
            num_employees, customer_interval, call_duration_mean, simulation_time, progress_placeholder, chart_placeholder
        )

        # Display the results
        st.write(f"### Average Wait Time: {avg_wait:.2f} units")
        st.write("### Final Queue Length and Employee Utilization Metrics:")

        # Plot the final metrics
        plot_final_metrics(queue_lengths, employee_utilization, simulation_time)

# This makes sure the app runs only when this file is executed directly
if __name__ == "__main__":
    show()
