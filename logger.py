import simpy
import numpy as np

class EventLogger:
    def __init__(self):
        self.event_log = []

    def log_event(self, customer_id, event_type, time):
        self.event_log.append({'customer_id': customer_id, 'event': event_type, 'time': time})

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.event_log)
