import time
import sys
from cogagent.core.metric.base_metric import BaseMetric
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
from cogagent.models.base_sclstm_model import LMDeep


class SclstmMetric(BaseMetric):
    def __init__(self, default_metric_name=None):
        super().__init__()
        self.default_metric_name = default_metric_name
        self.total_loss = 0
        self.redunt = 0
        self.miss = 0
        self.step = 0
        self.total = 0
        self.se = 0
        self.best_loss = 0
        if default_metric_name is None:
            self.default_metric_name = "total_loss"
        else:
            self.default_metric_name = default_metric_name

        # self.model = LMDeep()

    def evaluate(self, countBatch, countPerGen, loss):
        # self.total += countBatch['total']
        self.step = self.step + 1
        self.redunt += countBatch['redunt']
        self.miss += countBatch['miss']
        self.total += countBatch['total']
        self.total_loss += loss
        self.se = (self.redunt + self.miss) / self.total * 100
        # if self.total_loss/self.step > (self.total_loss+loss)/(self.step+1):
            # self.best_loss = (self.total_loss+loss)/(self.step+1)

    def get_metric(self, reset=True):
        redunt = self.redunt
        miss = self.miss
        total_loss = self.total_loss / self.step+1
        slot_error = self.se
        # best_loss = self.best_loss
        evaluate_result = {
            # "total": self.total,
            "redunt": redunt,
            "miss": miss,
            "total_loss": total_loss,
            "slot_error": slot_error,
            # "best_loss": best_loss,
        }
        if reset:
            self.step = 0
            self.redunt = 0
            self.miss = 0
            self.total = 0
            self.se = 0
            # self.best_loss = 0
            self.total_loss = 0
        return evaluate_result

