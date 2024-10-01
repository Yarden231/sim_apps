import streamlit as st
import simpy
import random
import statistics
import matplotlib.pyplot as plt
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
    
    # Make sure queue_lengths and employee_utilization match the number of time points
    queue_lengths = call_center.queue_lengths[:step + 1]
    employee_utilization = [util * 100 for util in call_center.employee_utilization[:step + 1]]

    # Ensure that lengths of both lists are the same as time_points
    if len(queue_lengths) < len(time_points):
        time_points = time_points[:len(queue_lengths)]
    if len(employee_utilization) < len(time_points):
        time_points = time_points[:len(employee_utilization)]

    # Plot the data
    plt.figure(figsize=(14, 4))
    plt.plot(time_points, queue_lengths, label='אורך תור (Queue Length)')
    plt.plot(time_points, employee_utilization, label='ניצולת עובדים (%)')
    plt.xlabel('זמן (Time)')
    plt.ylabel('ערך (Value)')
    plt.title(f'תור וניצולת עובדים בזמן אמת בשלב {step}')
    plt.legend()
    chart_placeholder.pyplot(plt.gcf())
    plt.clf()


def plot_final_metrics(queue_lengths, employee_utilization, simulation_time):
    time_points = list(range(simulation_time))

    # Plot Queue Length Over Time
    plt.figure(figsize=(14, 4))
    plt.plot(time_points, queue_lengths, label='אורך תור (Queue Length)')
    plt.xlabel('זמן (Time)')
    plt.ylabel('אורך תור (Queue Length)')
    plt.title('אורך התור לאורך זמן')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    # Plot Employee Utilization Over Time
    plt.figure(figsize=(14, 4))
    plt.plot(time_points, employee_utilization, label='ניצולת עובדים (%)')
    plt.xlabel('זמן (Time)')
    plt.ylabel('ניצולת עובדים (%)')
    plt.title('ניצולת עובדים לאורך זמן')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# Encapsulate Streamlit app in a function
def show():
    st.title("סימולציית מרכז שירות לקוחות בזמן אמת")

    # Slider inputs for simulation parameters (translated to Hebrew)
    num_employees = st.slider("מספר נציגי שירות", min_value=1, max_value=10, value=3)
    customer_interval = st.slider("זמן ממוצע בין הגעות לקוחות (בדקות)", min_value=1, max_value=10, value=4)
    call_duration_mean = st.slider("משך שיחה ממוצע (בדקות)", min_value=1, max_value=10, value=8)
    simulation_time = st.slider("זמן סימולציה (יחידות)", min_value=100, max_value=1000, value=400)

    # Display the selected parameters
    st.write("### פרמטרים נבחרים לסימולציה:")
    st.write(f"- מספר נציגי שירות: {num_employees}")
    st.write(f"- זמן ממוצע בין הגעות לקוחות: {customer_interval} דקות")
    st.write(f"- משך שיחה ממוצע: {call_duration_mean} דקות")
    st.write(f"- זמן סימולציה: {simulation_time} יחידות")

    # Placeholder for progress bar and real-time chart
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Run simulation when button is pressed
    if st.button("הפעל סימולציה"):
        avg_wait, queue_lengths, employee_utilization = run_simulation(
            num_employees, customer_interval, call_duration_mean, simulation_time, progress_placeholder, chart_placeholder
        )

        # Display the results
        st.write(f"### זמן המתנה ממוצע: {avg_wait:.2f} יחידות")
        st.write("### אורך התור וניצולת עובדים לאורך זמן")

        # Plot the final metrics
        plot_final_metrics(queue_lengths, employee_utilization, simulation_time)

# This makes sure the app runs only when this file is executed directly
if __name__ == "__main__":
    show()
