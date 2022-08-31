class BaseMetric:
    def __init__(self):
        pass

    def evaluate(self, pred, label):
        pass

    def get_metric(self, reset=True):
        pass