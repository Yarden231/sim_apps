import streamlit as st
import simpy
import numpy as np
import pandas as pd
import altair as alt
from utils import set_rtl

# Call the set_rtl function to apply RTL styles
set_rtl()

class Employee:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.busy = False

    def start_call(self):
        self.busy = True

    def end_call(self):
        self.busy = False

class CallCenter:
    def __init__(self, env, num_employees, support_time):
        self.env = env
        self.support_time = support_time
        self.queue = simpy.Resource(env, num_employees)
        self.employees = [Employee(env, i) for i in range(num_employees)]
        self.queue_lengths = []
        self.employee_status = []

    def log_queue_length(self):
        self.queue_lengths.append((self.env.now, len(self.queue.queue)))

    def support(self, customer, employee):
        employee.start_call()
        yield self.env.timeout(max(1, np.random.normal(self.support_time, 2)))
        employee.end_call()
        self.log_employee_status()

    def log_employee_status(self):
        status_snapshot = {'Time': self.env.now}
        for emp in self.employees:
            status_snapshot[f'Employee_{emp.id}'] = 1 if emp.busy else 0
        self.employee_status.append(status_snapshot)

def customer(env, name, call_center):
    call_center.log_queue_length()
    with call_center.queue.request() as request:
        yield request
        free_employee = next((e for e in call_center.employees if not e.busy), None)
        if free_employee:
            yield env.process(call_center.support(name, free_employee))
    call_center.log_queue_length()

def setup(env, call_center, customer_interval):
    i = 0
    while True:
        yield env.timeout(np.random.exponential(customer_interval))
        env.process(customer(env, f"Customer_{i}", call_center))
        i += 1

def run_simulation(num_employees, avg_support_time, customer_interval, sim_time):
    env = simpy.Environment()
    call_center = CallCenter(env, num_employees, avg_support_time)
    env.process(setup(env, call_center, customer_interval))
    env.run(until=sim_time)
    return call_center.queue_lengths, call_center.employee_status

def show():
    st.header("Call Center Simulation")
    
    num_employees = st.slider("Number of Employees", 1, 10, 2)
    avg_support_time = st.slider("Average Support Time (minutes)", 1, 20, 5)
    customer_interval = st.slider("Average Time Between Customers (minutes)", 1, 10, 2)
    sim_time = st.slider("Simulation Time (minutes)", 60, 600, 120)

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            queue_lengths, employee_status = run_simulation(num_employees, avg_support_time, customer_interval, sim_time)

        # Display results
        queue_df = pd.DataFrame(queue_lengths, columns=['Time', 'Queue Length'])
        status_df = pd.DataFrame(employee_status)

        st.subheader("Queue Length Over Time")
        queue_chart = alt.Chart(queue_df).mark_line().encode(
            x='Time',
            y='Queue Length',
            tooltip=['Time', 'Queue Length']
        ).interactive()
        st.altair_chart(queue_chart, use_container_width=True)

        st.subheader("Employee Status Over Time")
        status_melted = status_df.melt(id_vars=['Time'], var_name='Employee', value_name='Is Busy')
        status_chart = alt.Chart(status_melted).mark_line(interpolate='step-after').encode(
            x='Time',
            y='Is Busy',
            color='Employee',
            tooltip=['Time', 'Employee', 'Is Busy']
        ).interactive()
        st.altair_chart(status_chart, use_container_width=True)

    st.subheader("Learning Objectives")
    st.write("""
    - Understand how to model a multi-server queueing system
    - Analyze the impact of different parameters on system performance
    - Interpret simulation results and make data-driven decisions
    """)
