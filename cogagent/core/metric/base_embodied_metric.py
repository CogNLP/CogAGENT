from cogagent.core.metric.base_metric import BaseMetric
import torch
import numpy as np


class BaseEnbodiedMetric(BaseMetric):

    def __init__(self,
                 default_metric_name):
        super().__init__()
        self.default_metric_name = default_metric_name
        self.pre_list = {}
        self.pre_list['pred'] = []

    def accuracy(self, dists, threshold=3):
        """Calculating accuracy at 3 meters by default"""
        return np.mean((torch.tensor(dists) <= threshold).int().numpy())

    def evaluate(self, pred, label=None):
        self.pre_list['pred'].extend(pred)

    def get_metric(self, reset=True):
        evaluate_result = {}
        acc0m = self.accuracy(self.pre_list['pred'], 0)
        acc5m = self.accuracy(self.pre_list['pred'], 5)

        evaluate_result = {'acc0m': acc0m,
                           'acc5m': acc5m,
                           }

        if reset:
            self.pre_list = {}
            self.pre_list['pred'] = []
        return evaluate_result
