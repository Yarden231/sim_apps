import simpy
import numpy as np

class FoodTruck:
    def __init__(self, env, order_time_min, order_time_max, config, logger):
        self.env = env
        self.logger = logger
        self.order_station = simpy.Resource(env, capacity=config['order_capacity'])
        self.prep_station = simpy.Resource(env, capacity=config['prep_capacity'])
        self.pickup_station = simpy.Resource(env, capacity=config['pickup_capacity'])
        self.batch = []
        self.left_count = 0
        self.left_before_ordering = 0
        self.total_visitors = 0
        self.undercooked_count = 0
        self.wait_times = {'order': [], 'prep': [], 'pickup': [], 'total': []}
        self.queue_sizes = {'order': [], 'prep': [], 'pickup': [], 'total': []}
        self.left_over_time = []
        self.left_before_ordering_over_time = []
        self.undercooked_over_time = []
        self.total_visitors_over_time = []
        self.order_time_min = order_time_min
        self.order_time_max = order_time_max

    def process_service(self, station, visitor, service_time_range):
        with station.request() as req:
            yield req
            start_time = self.env.now
            service_time = np.random.uniform(*service_time_range)
            yield self.env.timeout(service_time)
        end_time = self.env.now
        return start_time, end_time

    def order_service(self, visitor):
        arrival_time = self.env.now
        self.logger.log_event(visitor['name'], 'arrival', arrival_time)
        start_time, end_time = yield from self.process_service(self.order_station, visitor, (self.order_time_min, self.order_time_max))
        self.wait_times['order'].append(end_time - arrival_time)
        self.logger.log_event(visitor['name'], 'order_complete', end_time)
        return end_time - start_time

    def prep_service(self, visitors):
        prep_start = self.env.now
        with self.prep_station.request() as req:
            yield req
            for visitor in visitors:
                self.logger.log_event(visitor['name'], 'preparing', self.env.now)
            service_time = np.random.normal(5, 1)
            yield self.env.timeout(service_time)
        prep_end = self.env.now
        undercooked = np.random.rand(len(visitors)) < 0.3
        self.undercooked_count += sum(undercooked)
        for i, visitor in enumerate(visitors):
            visitor['prep_time'] = prep_end - prep_start
            self.wait_times['prep'].append(visitor['prep_time'])
            self.logger.log_event(visitor['name'], 'prep_complete', prep_end)
            if undercooked[i]:
                self.logger.log_event(visitor['name'], 'undercooked', prep_end)
        for visitor in visitors:
            self.env.process(self.pickup_service(visitor))

    def pickup_service(self, visitor):
        pickup_start = self.env.now
        start_time, end_time = yield from self.process_service(self.pickup_station, visitor, (2, 4))
        self.logger.log_event(visitor['name'], 'exit', end_time)
        pickup_time = end_time - pickup_start
        self.wait_times['pickup'].append(pickup_time)
        total_time = end_time - visitor['arrival_time']
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
            total_queue_size = len(self.order_station.queue) + len(self.prep_station.queue) + len(self.pickup_station.queue)
            self.queue_sizes['total'].append(total_queue_size)
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

def run_simulation(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability, config, logger):
    env = simpy.Environment()
    food_truck = FoodTruck(env, order_time_min, order_time_max, config, logger)
    env.process(arrival_process(env, food_truck, arrival_rate, leave_probability))
    env.process(food_truck.monitor())
    env.run(until=sim_time)
    return food_truck

def run_simulation_with_speed(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability, config, logger, speed):
    env = simpy.Environment()
    food_truck = FoodTruck(env, order_time_min, order_time_max, config, logger)
    env.process(arrival_process(env, food_truck, arrival_rate, leave_probability))
    env.process(food_truck.monitor())
    while env.now < sim_time:
        env.step()
        env.timeout(1 / speed)
    return food_truck
