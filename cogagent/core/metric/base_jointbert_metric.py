import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class BaseJointbertMetric(BaseMetric):
    def __init__(self, mode, default_metric_name=None):
        super().__init__()
        if mode not in ["binary", "multi"]:
            raise ValueError("Please choose mode in binary or multi")
        self.mode = mode
        self.label_list = list()
        self.pre_list = list()
        self.default_metric_name = default_metric_name
        if default_metric_name is None:
            self.default_metric_name = "F1" if mode == "binary" else "macro_F1"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, pred, labels):
        
        # self.label_list.append(labels)
        # print(self.label_list)
        self.label_list = self.label_list + labels
        self.pre_list = self.pre_list + pred
        # self.pre_list.append(pred)
        # print(self.pre_list)

    def get_metric(self, reset=True):
        evaluate_result = {}
        if self.mode == "binary":
            TP, FP, FN = 0, 0, 0
            predicts = self.pre_list
            labels = self.label_list
            for ele in predicts:
                if ele in labels:
                    TP += 1
                else:
                    FP += 1
            for ele in labels:
                if ele not in predicts:
                    FN += 1
            # print(TP, FP, FN)
            precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
            recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
            F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
            # P = precision_score(self.label_list, self.pre_list, average="binary")
            # R = recall_score(self.label_list, self.pre_list, average="binary")
            # F1 = f1_score(self.label_list, self.pre_list, average="binary")
            # Acc = accuracy_score(self.label_list, self.pre_list)
            # Acc = accuracy_score(label, pred)
            evaluate_result = {
                                "P": precision,
                               "R": recall,
                               "F1": F1,
                            #    "Acc": Acc,
                               }
        if self.mode == "multi":
            micro_P = precision_score(self.label_list, self.pre_list, average="micro")
            micro_R = recall_score(self.label_list, self.pre_list, average="micro")
            micro_F1 = f1_score(self.label_list, self.pre_list, average="micro")
            macro_P = precision_score(self.label_list, self.pre_list, average="macro")
            macro_R = recall_score(self.label_list, self.pre_list, average="macro")
            macro_F1 = f1_score(self.label_list, self.pre_list, average="macro")
            Acc = accuracy_score(self.label_list, self.pre_list)
            evaluate_result = {"micro_P": micro_P,
                               "micro_R": micro_R,
                               "micro_F1": micro_F1,
                               "macro_P": macro_P,
                               "macro_R": macro_R,
                               "macro_F1": macro_F1,
                               "Acc": Acc,
                               }
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result
