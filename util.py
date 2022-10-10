import time

class TimeArr():
    # Time things
    def __init__(self, auto_start=True):
        self.timestamps = []
        self.labels = []
        if auto_start:
            self.start()
    
    def start(self):
        self.save("Start")

    def save(self, label=None):
        self.timestamps.append(time.time())
        self.labels.append(label if label is not None else len(self.timestamps))
    
    def last(self):
        i = len(self.timestamps) - 1
        t_prev = self.timestamps[i-1] if i > 0 else self.timestamps[i]
        return self.timestamps[i] - t_prev
    
    def get_intervals(self):
        intervals = dict()
        for i in range(len(self.timestamps)):
            t_prev = self.timestamps[i-1] if i > 0 else self.timestamps[i]
            intervals[self.labels[i]] = self.timestamps[i] - t_prev
        return intervals

    def report(self, print_func=print):
        for label, length in self.get_intervals().items():
            print_func(f'{label} -> {length:.2f}')