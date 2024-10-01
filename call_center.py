import streamlit as st
import simpy
import random
import statistics
import matplotlib.pyplot as plt
import time

# Simulation Classes and Functions (unchanged)
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
        # Update real-time progress and graph
        update_real_time_chart(call_center, i, chart_placeholder)
        progress_placeholder.progress((i + 1) / simulation_time)
        time.sleep(0.01)  # Sleep to mimic real-time progress

    # Return the results after the simulation completes
    if call_center.wait_times:
        avg_wait = statistics.mean(call_center.wait_times)
    else:
        avg_wait = 0

    employee_utilization_percentages = [util * 100 for util in call_center.employee_utilization]

    return avg_wait, call_center.queue_lengths, employee_utilization_percentages

def update_real_time_chart(call_center, step, chart_placeholder):
    # Plot real-time queue length and employee utilization
    time_points = list(range(step + 1))
    
    # Real-time queue length
    queue_lengths = call_center.queue_lengths[:step+1]
    employee_utilization = [util * 100 for util in call_center.employee_utilization[:step+1]]

    plt.figure(figsize=(14, 4))
    plt.plot(time_points, queue_lengths, label='Queue Length')
    plt.plot(time_points, employee_utilization, label='Employee Utilization (%)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Real-Time Queue Length and Employee Utilization at Step {step}')
    plt.legend()
    chart_placeholder.pyplot(plt.gcf())
    plt.clf()

def plot_final_metrics(queue_lengths, employee_utilization, simulation_time):
    time_points = list(range(simulation_time))

    # Plot Queue Length Over Time
    plt.figure(figsize=(14, 4))
    plt.plot(time_points, queue_lengths, label='Queue Length')
    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.title('Queue Length Over Time')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    # Plot Employee Utilization Over Time
    plt.figure(figsize=(14, 4))
    plt.plot(time_points, employee_utilization, label='Employee Utilization (%)')
    plt.xlabel('Time')
    plt.ylabel('Utilization (%)')
    plt.title('Employee Utilization Over Time')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# Encapsulate Streamlit app in a function
def show():
    st.title("Call Center Simulation")

    # Slider inputs for simulation parameters
    num_employees = st.slider("Number of Employees", min_value=1, max_value=10, value=3)
    customer_interval = st.slider("Customer Arrival Rate (mean interval)", min_value=1, max_value=10, value=4)
    call_duration_mean = st.slider("Call Duration Mean", min_value=1, max_value=10, value=8)
    simulation_time = st.slider("Simulation Time (units)", min_value=100, max_value=1000, value=400)

    # Display the selected parameters
    st.write("### Selected Simulation Parameters:")
    st.write(f"- Number of Employees: {num_employees}")
    st.write(f"- Customer Arrival Rate (mean interval): {customer_interval} units")
    st.write(f"- Call Duration Mean: {call_duration_mean} units")
    st.write(f"- Simulation Time: {simulation_time} units")

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
        st.write("### Final Queue Length and Employee Utilization Over Time")

        # Plot the final metrics
        plot_final_metrics(queue_lengths, employee_utilization, simulation_time)

# This makes sure the app runs only when this file is executed directly
if __name__ == "__main__":
    show()
