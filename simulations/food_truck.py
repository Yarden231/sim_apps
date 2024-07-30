import streamlit as st
import simpy
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, expon
import graphviz

# Food Truck Simulation
class FoodTruck:
    def __init__(self, env):
        self.env = env
        self.order_station = simpy.Resource(env, capacity=1)
        self.prep_station = simpy.Resource(env, capacity=1)
        self.pickup_station = simpy.Resource(env, capacity=1)
        self.batch = []
        self.left_count = 0
        self.left_before_ordering = 0
        self.total_visitors = 0
        self.undercooked_count = 0
        self.wait_times = {'order': [], 'prep': [], 'pickup': [], 'total': []}
        self.queue_sizes = {'order': [], 'prep': [], 'pickup': []}
        self.left_over_time = []
        self.left_before_ordering_over_time = []
        self.undercooked_over_time = []
        self.total_visitors_over_time = []

    def order_service(self, visitor):
        arrival_time = self.env.now
        with self.order_station.request() as req:
            yield req
            order_start = self.env.now
            service_time = np.random.uniform(3, 7)
            yield self.env.timeout(service_time)
        order_end = self.env.now
        self.wait_times['order'].append(order_end - arrival_time)
        return order_end - order_start

    def prep_service(self, visitors):
        prep_start = self.env.now
        with self.prep_station.request() as req:
            yield req
            service_time = np.random.normal(5, 1)
            yield self.env.timeout(service_time)
        prep_end = self.env.now
        undercooked = np.random.rand(len(visitors)) < 0.3
        self.undercooked_count += sum(undercooked)
        for visitor in visitors:
            visitor['prep_time'] = prep_end - prep_start
            self.wait_times['prep'].append(visitor['prep_time'])
            yield self.env.process(self.pickup_service(visitor))

    def pickup_service(self, visitor):
        pickup_start = self.env.now
        with self.pickup_station.request() as req:
            yield req
            service_time = np.random.uniform(2, 4)
            yield self.env.timeout(service_time)
        pickup_end = self.env.now
        pickup_time = pickup_end - pickup_start
        self.wait_times['pickup'].append(pickup_time)
        total_time = pickup_end - visitor['arrival_time']
        self.wait_times['total'].append(total_time)

    def process_batch(self):
        while self.batch:
            batch_size = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            visitors_to_process = self.batch[:batch_size]
            del self.batch[:batch_size]
            yield self.env.process(self.prep_service(visitors_to_process))

    def monitor(self):
        while True:
            self.queue_sizes['order'].append(len(self.order_station.queue))
            self.queue_sizes['prep'].append(len(self.prep_station.queue))
            self.queue_sizes['pickup'].append(len(self.pickup_station.queue))
            self.left_over_time.append(self.left_count)
            self.left_before_ordering_over_time.append(self.left_before_ordering)
            self.undercooked_over_time.append(self.undercooked_count)
            self.total_visitors_over_time.append(self.total_visitors)
            yield self.env.timeout(1)

def visitor(env, name, food_truck):
    food_truck.total_visitors += 1
    arrival_time = env.now

    if np.random.random() < 0.1:
        food_truck.left_before_ordering += 1
        food_truck.left_count += 1
        return

    order_time = yield env.process(food_truck.order_service({'name': name, 'arrival_time': arrival_time}))
    food_truck.batch.append({'name': name, 'arrival_time': arrival_time, 'order_time': order_time})
    if len(food_truck.batch) >= 1:
        env.process(food_truck.process_batch())

def arrival_process(env, food_truck):
    visitor_count = 0
    while True:
        yield env.timeout(np.random.exponential(10))
        visitor_count += 1
        env.process(visitor(env, visitor_count, food_truck))

def run_simulation(sim_time=10000):
    env = simpy.Environment()
    food_truck = FoodTruck(env)
    env.process(arrival_process(env, food_truck))
    env.process(food_truck.monitor())
    env.run(until=sim_time)
    return food_truck

def create_flowchart():
    dot = graphviz.Digraph(comment='The Busy Food Truck Flow')
    dot.attr(rankdir='TB', size='8,8')

    # Customer Arrival
    dot.node('A', 'Customer Arrival\n(Exp. dist, mean 10 min)')

    # Order Station
    dot.node('C', 'Order Station')
    dot.edge('A', 'Leave', label='10% Leave')
    dot.edge('A', 'C', label='90% Stay')
    
    # Order Time
    dot.node('D1', 'A: 3-4 min\n(Uniform)')
    dot.node('D2', 'B: 4-6 min\n(Triangular)')
    dot.node('D3', 'C: 10 min\n(Fixed)')
    dot.edge('C', 'D1', label='50%')
    dot.edge('C', 'D2', label='25%')
    dot.edge('C', 'D3', label='25%')

    # Meal Preparation
    dot.node('E', 'Meal Preparation\n(Normal dist, μ=5, σ=1)')
    dot.edge('D1', 'E')
    dot.edge('D2', 'E')
    dot.edge('D3', 'E')

    # Batch Size
    dot.node('F1', 'Individual')
    dot.node('F2', 'Batch of 2')
    dot.node('F3', 'Batch of 3')
    dot.edge('E', 'F1', label='20%')
    dot.edge('E', 'F2', label='50%')
    dot.edge('E', 'F3', label='30%')

    # Cooking Quality (for Batch of 3)
    dot.node('G1', 'Properly Cooked')
    dot.node('G2', 'Undercooked')
    dot.edge('F3', 'G1', label='70%')
    dot.edge('F3', 'G2', label='30%')

    # Pickup Station
    dot.node('H', 'Pickup Station\n(Uniform 2-4 min)')
    dot.edge('F1', 'H')
    dot.edge('F2', 'H')
    dot.edge('G1', 'H')
    dot.edge('G2', 'H')

    # Customer Departure
    dot.node('I', 'Leave satisfied')
    dot.edge('H', 'I')

    return dot

def plot_sampling_comparison(composition_samples, accept_reject_samples):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram with KDE
    sns.histplot(composition_samples, kde=True, label='Composition Samples', color='green', stat="density", ax=axes[0, 0], alpha=0.6)
    sns.histplot(accept_reject_samples, kde=True, label='Accept-Reject Samples', color='orange', stat="density", ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title('Histogram and KDE of Samples')
    axes[0, 0].set_xlabel('Service Time')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # Empirical CDF
    composition_sorted = np.sort(composition_samples)
    accept_reject_sorted = np.sort(accept_reject_samples)
    composition_cdf = np.arange(1, len(composition_sorted) + 1) / len(composition_sorted)
    accept_reject_cdf = np.arange(1, len(accept_reject_sorted) + 1) / len(accept_reject_sorted)
    
    axes[0, 1].step(composition_sorted, composition_cdf, label='Composition Samples', color='green')
    axes[0, 1].step(accept_reject_sorted, accept_reject_cdf, label='Accept-Reject Samples', color='orange')
    axes[0, 1].set_title('Empirical CDF of Samples')
    axes[0, 1].set_xlabel('Service Time')
    axes[0, 1].set_ylabel('CDF')
    axes[0, 1].legend()

    # QQ Plot
    probplot(composition_samples, dist="uniform", sparams=(3, 7), plot=axes[1, 0])
    probplot(accept_reject_samples, dist="uniform", sparams=(3, 7), plot=axes[1, 0])
    axes[1, 0].set_title('QQ Plot - Composition and Accept-Reject Samples')
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Sample Quantiles')

    # Summary Statistics
    axes[1, 1].axis('off')
    textstr = '\n'.join((
        f'Composition Samples Mean: {np.mean(composition_samples):.2f}',
        f'Composition Samples Variance: {np.var(composition_samples):.2f}',
        f'Accept-Reject Samples Mean: {np.mean(accept_reject_samples):.2f}',
        f'Accept-Reject Samples Variance: {np.var(accept_reject_samples):.2f}'
    ))
    axes[1, 1].text(0.05, 0.5, textstr, transform=axes[1, 1].transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    axes[1, 1].set_title('Summary Statistics')

    plt.tight_layout()
    return fig

def plot_simulation_results(food_truck):
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    time = np.arange(len(food_truck.queue_sizes['order']))

    # Queue sizes at each station over time
    for station in ['order', 'prep', 'pickup']:
        axes[0, 0].plot(time, food_truck.queue_sizes[station], label=f'{station.capitalize()} Station')
    axes[0, 0].set_title('Queue Sizes at Each Station Over Time')
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Queue Size')
    axes[0, 0].legend()

    # Distribution of waiting times
    for wait_type in ['order', 'prep', 'pickup', 'total']:
        axes[0, 1].hist(food_truck.wait_times[wait_type], bins=20, alpha=0.5, label=f'{wait_type.capitalize()} Time')
    axes[0, 1].set_title('Distribution of Waiting Times')
    axes[0, 1].set_xlabel('Waiting Time (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Visitors who left without ordering
    axes[0, 2].plot(time, food_truck.left_over_time, color='orange', label='Total Left')
    axes[0, 2].plot(time, food_truck.left_before_ordering_over_time, color='red', label='Left Before Ordering')
    axes[0, 2].set_title('Visitors Who Left Over Time')
    axes[0, 2].set_xlabel('Time (minutes)')
    axes[0, 2].set_ylabel('Number of Visitors')
    axes[0, 2].legend()

    # Undercooked meals
    axes[0, 3].plot(time, food_truck.undercooked_over_time, color='red')
    axes[0, 3].set_title('Undercooked Meals Over Time')
    axes[0, 3].set_xlabel('Time (minutes)')
    axes[0, 3].set_ylabel('Number of Undercooked Meals')

    # Total visitors over time
    axes[1, 0].plot(time, food_truck.total_visitors_over_time, color='green')
    axes[1, 0].set_title('Total Visitors Over Time')
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Number of Visitors')

# Continuing from the previous code...

    # Average waiting times over time
    window = 50  # Window size for moving average
    for wait_type in ['order', 'prep', 'pickup', 'total']:
        avg_wait = np.convolve(food_truck.wait_times[wait_type], np.ones(window), 'valid') / window
        axes[1, 1].plot(range(window-1, len(avg_wait) + window-1), avg_wait, label=f'{wait_type.capitalize()} Time')
    axes[1, 1].set_title('Average Waiting Times Over Time')
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Average Waiting Time (minutes)')
    axes[1, 1].legend()

    # Percentage of customers who left over time
    epsilon = 1e-10  # Small value to avoid division by zero
    total_left_percentage = np.array(food_truck.left_over_time) / (np.array(food_truck.total_visitors_over_time) + epsilon) * 100
    left_before_ordering_percentage = np.array(food_truck.left_before_ordering_over_time) / (np.array(food_truck.total_visitors_over_time) + epsilon) * 100
    axes[1, 2].plot(time, total_left_percentage, color='orange', label='Total Left')
    axes[1, 2].plot(time, left_before_ordering_percentage, color='red', label='Left Before Ordering')
    axes[1, 2].set_title('Percentage of Customers Who Left Over Time')
    axes[1, 2].set_xlabel('Time (minutes)')
    axes[1, 2].set_ylabel('Percentage of Customers')
    axes[1, 2].legend()

    # Queue Size Distribution
    for station in ['order', 'prep', 'pickup']:
        if len(set(food_truck.queue_sizes[station])) > 1:  # Ensure there is variance
            sns.kdeplot(food_truck.queue_sizes[station], ax=axes[1, 3], fill=True, label=f'{station.capitalize()} Station')
    axes[1, 3].set_title('Distribution of Queue Sizes')
    axes[1, 3].set_xlabel('Queue Size')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].legend()

    # Waiting Time Correlation
    min_length = min(len(food_truck.wait_times['order']),
                     len(food_truck.wait_times['prep']),
                     len(food_truck.wait_times['pickup']))
    wait_times = np.array([food_truck.wait_times['order'][:min_length],
                           food_truck.wait_times['prep'][:min_length],
                           food_truck.wait_times['pickup'][:min_length]])
    sns.heatmap(np.corrcoef(wait_times), annot=True, cmap='coolwarm',
                xticklabels=['Order', 'Prep', 'Pickup'],
                yticklabels=['Order', 'Prep', 'Pickup'], ax=axes[2, 0])
    axes[2, 0].set_title('Correlation between Waiting Times')

    # Customer Flow
    labels = ['Served', 'Left Before Ordering', 'Left After Ordering']
    sizes = [food_truck.total_visitors - food_truck.left_count,
             food_truck.left_before_ordering,
             food_truck.left_count - food_truck.left_before_ordering]
    axes[2, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[2, 1].set_title('Customer Flow')

    # Cumulative Metrics
    axes[2, 2].plot(np.cumsum(food_truck.total_visitors_over_time), label='Total Visitors')
    axes[2, 2].plot(np.cumsum(food_truck.left_before_ordering_over_time), label='Left Before Ordering')
    axes[2, 2].plot(np.cumsum(food_truck.undercooked_over_time), label='Undercooked Meals')
    axes[2, 2].set_title('Cumulative Metrics Over Time')
    axes[2, 2].set_xlabel('Time (minutes)')
    axes[2, 2].set_ylabel('Count')
    axes[2, 2].legend()

    # Average Metrics
    axes[2, 3].plot(np.cumsum(food_truck.total_visitors_over_time)/np.arange(1, len(food_truck.total_visitors_over_time)+1), label='Average Visitors')
    axes[2, 3].plot(np.cumsum(food_truck.left_before_ordering_over_time)/np.arange(1, len(food_truck.left_before_ordering_over_time)+1), label='Average Left Before Ordering')
    axes[2, 3].plot(np.cumsum(food_truck.undercooked_over_time)/np.arange(1, len(food_truck.undercooked_over_time)+1), label='Average Undercooked Meals')
    axes[2, 3].set_title('Average Metrics Over Time')
    axes[2, 3].set_xlabel('Time (minutes)')
    axes[2, 3].set_ylabel('Average Count')
    axes[2, 3].legend()

    # Customer Satisfaction Proxy
    satisfaction = np.array(food_truck.wait_times['total']) < 30  # Assuming satisfaction if total wait < 30 mins
    labels = ['Satisfied', 'Unsatisfied']
    sizes = [np.sum(satisfaction), len(satisfaction) - np.sum(satisfaction)]
    axes[3, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[3, 0].set_title('Customer Satisfaction Proxy')

    # Undercooked Meals Rate
    undercooked_rate = [u / max(1, t) for u, t in zip(food_truck.undercooked_over_time, food_truck.total_visitors_over_time)]
    axes[3, 1].plot(undercooked_rate)
    axes[3, 1].set_title('Undercooked Meals Rate Over Time')
    axes[3, 1].set_xlabel('Time (minutes)')
    axes[3, 1].set_ylabel('Undercooked Rate')

    # Efficiency Metric
    efficiency = [t / max(1, (t + l)) for t, l in zip(food_truck.total_visitors_over_time, food_truck.left_over_time)]
    axes[3, 2].plot(efficiency)
    axes[3, 2].set_title('Efficiency Metric Over Time')
    axes[3, 2].set_xlabel('Time (minutes)')
    axes[3, 2].set_ylabel('Efficiency (Served / Total Arrived)')

    # Waiting Time Distribution Comparison
    for wait_type in ['order', 'prep', 'pickup', 'total']:
        if len(food_truck.wait_times[wait_type]) > 1:  # Ensure there is data to plot
            sns.kdeplot(food_truck.wait_times[wait_type], ax=axes[3, 3], fill=True, label=f'{wait_type.capitalize()} Time')
    axes[3, 3].set_title('Waiting Time Distribution Comparison')
    axes[3, 3].set_xlabel('Waiting Time (minutes)')
    axes[3, 3].set_ylabel('Density')
    axes[3, 3].legend()

    plt.tight_layout()
    return fig

def show():
    st.title("The Busy Food Truck Simulation")

    st.write("""
    Welcome to "The Busy Food Truck" Experience! 🍔🚚
    
    Explore the dynamics of street food service through this interactive simulation 
    based on queuing theory and probability distributions.
    """)

    st.header("Simulation Flow")
    flowchart = create_flowchart()
    st.graphviz_chart(flowchart)

    st.header("Sampling Methods Comparison")
    n_samples = st.slider("Number of samples", 1000, 10000, 5000)
    composition_samples = composition_sampling(n_samples)
    accept_reject_samples = accept_reject_sampling(n_samples)
    
    fig_sampling = plot_sampling_comparison(composition_samples, accept_reject_samples)
    st.pyplot(fig_sampling)

    st.header("Food Truck Simulation")
    sim_time = st.slider("Simulation time (minutes)", 1000, 10000, 5000)
    
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            food_truck = run_simulation(sim_time)
        
        st.success("Simulation complete!")

        st.subheader("Simulation Results")
        st.write(f"Total visitors: {food_truck.total_visitors}")
        st.write(f"Visitors who left (total): {food_truck.left_count}")
        st.write(f"Visitors who left before ordering: {food_truck.left_before_ordering}")
        st.write(f"Undercooked meals: {food_truck.undercooked_count}")
        
        for wait_type in ['order', 'prep', 'pickup', 'total']:
            avg_wait = np.mean(food_truck.wait_times[wait_type])
            st.write(f"Average {wait_type} time: {avg_wait:.2f} minutes")
        
        st.write(f"Percentage of visitors who left: {(food_truck.left_count / food_truck.total_visitors) * 100:.2f}%")
        st.write(f"Percentage of visitors who left before ordering: {(food_truck.left_before_ordering / food_truck.total_visitors) * 100:.2f}%")

        fig_results = plot_simulation_results(food_truck)
        st.pyplot(fig_results)

    st.write("""
    #### Engage and Explore
    Experience the operational challenges and enjoy the excitement of running a street food truck 
    through this interactive simulation. Adjust the parameters and run multiple simulations to see 
    how different factors affect the food truck's performance!
    """)

if __name__ == "__main__":
    show()