from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from cogagent.models.sclstm.model.lm_deep import LMDeep
from cogagent.utils.KE_Blender_utils import bleu_metric
from cogagent.utils.KE_Blender_utils import f_one


class KE_BlenderMetric(BaseMetric):
    def __init__(self, default_metric_name=None):
        super().__init__()
        self.default_metric_name = default_metric_name
        self.correct = 0
        self.count = 0
        self.Acc = 0.0
        self.step = 1
        self.valid_loss = 0.0
        self.gold = []
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.res = 0.0
        if default_metric_name is None:
            self.default_metric_name = "Acc"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, dev_loss, correct, count, words, dev_labels):
        # self.total += countBatch['total']
        self.step = self.step + 1
        self.correct += correct
        self.count += count
        self.valid_loss += dev_loss
        self.Acc = self.correct / self.count
        self.b1, self.b2, self.b3 = bleu_metric(dev_labels, words)
        self.res = f_one(dev_labels, words)

    def get_metric(self, reset=True):

        valid_loss = self.valid_loss / self.step + 1
        Acc = self.Acc
        B1 = self.b1
        B2 = self.b2
        B3 = self.b3
        f1 = self.res[0]
        pre = self.res[1]
        rec = self.res[2]
        evaluate_result = {
            "valid_loss": valid_loss,
            "Acc": Acc,
            "B1": B1,
            "B2": B2,
            "B3": B3,
            "f1": f1,
            "pre": pre,
            "rec": rec,
        }
        if reset:
            self.step = 0
            self.Acc = 0.0
            self.valid_loss = 0
            self.b1 = 0
            self.b2 = 0
            self.b3 = 0
            self.res = 0.0
        return evaluate_result

