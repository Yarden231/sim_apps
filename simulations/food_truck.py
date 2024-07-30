import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FoodTruck:
    def __init__(self, env, order_time_min, order_time_max):
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
        self.order_time_min = order_time_min
        self.order_time_max = order_time_max

    def order_service(self, visitor):
        arrival_time = self.env.now
        with self.order_station.request() as req:
            yield req
            order_start = self.env.now
            service_time = np.random.uniform(self.order_time_min, self.order_time_max)
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

def visitor(env, name, food_truck, leave_probability):
    food_truck.total_visitors += 1
    arrival_time = env.now

    if np.random.random() < leave_probability:
        food_truck.left_before_ordering += 1
        food_truck.left_count += 1
        return

    order_time = yield env.process(food_truck.order_service({'name': name, 'arrival_time': arrival_time}))
    food_truck.batch.append({'name': name, 'arrival_time': arrival_time, 'order_time': order_time})
    if len(food_truck.batch) >= 1:
        env.process(food_truck.process_batch())

def arrival_process(env, food_truck, arrival_rate, leave_probability):
    visitor_count = 0
    while True:
        yield env.timeout(np.random.exponential(arrival_rate))
        visitor_count += 1
        env.process(visitor(env, visitor_count, food_truck, leave_probability))

def run_simulation(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability):
    env = simpy.Environment()
    food_truck = FoodTruck(env, order_time_min, order_time_max)
    env.process(arrival_process(env, food_truck, arrival_rate, leave_probability))
    env.process(food_truck.monitor())
    env.run(until=sim_time)
    return food_truck

def plot_simulation_results(food_truck):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    time = np.arange(len(food_truck.queue_sizes['order']))

    # Queue sizes over time
    for station in ['order', 'prep', 'pickup']:
        axes[0, 0].plot(time, food_truck.queue_sizes[station], label=f'{station.capitalize()} Station')
    axes[0, 0].set_title('Queue Sizes Over Time')
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Queue Size')
    axes[0, 0].legend()

    # Waiting time distribution
    for wait_type in ['order', 'prep', 'pickup', 'total']:
        sns.histplot(food_truck.wait_times[wait_type], kde=True, label=f'{wait_type.capitalize()}', ax=axes[0, 1])
    axes[0, 1].set_title('Waiting Time Distribution')
    axes[0, 1].set_xlabel('Waiting Time (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Visitors who left over time
    axes[1, 0].plot(time, food_truck.left_over_time, label='Total Left')
    axes[1, 0].plot(time, food_truck.left_before_ordering_over_time, label='Left Before Ordering')
    axes[1, 0].set_title('Visitors Who Left Over Time')
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Number of Visitors')
    axes[1, 0].legend()

    # Undercooked meals over time
    axes[1, 1].plot(time, food_truck.undercooked_over_time)
    axes[1, 1].set_title('Undercooked Meals Over Time')
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Number of Undercooked Meals')

    plt.tight_layout()
    return fig

def show():
    st.title("סימולציית משאית המזון העסוקה")

    st.write("""
    ברוכים הבאים לחוויית "משאית המזון העסוקה"! 🍔🚚
    
    חקרו את הדינמיקה של שירות מזון רחוב באמצעות סימולציה אינטראקטיבית זו 
    המבוססת על תורת התורים והתפלגויות הסתברותיות.
    """)

    # פרמטרים אינטראקטיביים
    st.sidebar.header("הגדרות סימולציה")
    sim_time = st.sidebar.slider("זמן סימולציה (דקות)", 1000, 10000, 5000)
    arrival_rate = st.sidebar.slider("זמן ממוצע בין הגעות לקוחות (דקות)", 5, 20, 10)
    order_time_min = st.sidebar.slider("זמן הזמנה מינימלי (דקות)", 1, 5, 3)
    order_time_max = st.sidebar.slider("זמן הזמנה מקסימלי (דקות)", 5, 10, 7)
    leave_probability = st.sidebar.slider("הסתברות לעזיבה לפני הזמנה", 0.0, 0.5, 0.1)

    if st.sidebar.button("הפעל סימולציה"):
        with st.spinner("מריץ סימולציה..."):
            food_truck = run_simulation(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability)
        
        st.success("הסימולציה הושלמה!")

        # תוצאות עיקריות
        col1, col2, col3 = st.columns(3)
        col1.metric("סך הכל מבקרים", f"{food_truck.total_visitors}")
        col2.metric("מבקרים שעזבו", f"{food_truck.left_count}")
        col3.metric("ארוחות לא מבושלות", f"{food_truck.undercooked_count}")

        # Average waiting times
        st.subheader("Average Waiting Times")
        wait_times_df = pd.DataFrame({
            'Station': ['Order', 'Preparation', 'Pickup', 'Total'],
            'Average Waiting Time (minutes)': [
                np.mean(food_truck.wait_times['order']),
                np.mean(food_truck.wait_times['prep']),
                np.mean(food_truck.wait_times['pickup']),
                np.mean(food_truck.wait_times['total'])
            ]
        })
        st.bar_chart(wait_times_df.set_index('Station'))

        # גרפים
        st.subheader("ניתוח הסימולציה")
        fig = plot_simulation_results(food_truck)
        st.pyplot(fig)

    st.write("""
    #### התנסו וחקרו
    חוו את האתגרים התפעוליים והתרגשות של ניהול משאית מזון רחוב 
    באמצעות סימולציה אינטראקטיבית זו. התאימו את הפרמטרים והריצו מספר סימולציות 
    כדי לראות כיצד גורמים שונים משפיעים על ביצועי משאית המזון!
    """)

if __name__ == "__main__":
    show()