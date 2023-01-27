from cogagent.core.metric.base_metric import BaseMetric
import torch


class BaseTopKMetric(BaseMetric):
    def __init__(self, topk, default_metric_name=None):
        super().__init__()
        if not isinstance(topk, list):
            raise ValueError("Please input topk in a list")
        self.topk = topk
        self.cor1_list = list()
        self.tot1_list = list()
        self.cor2_list = list()
        self.tot2_list = list()
        self.cor5_list = list()
        self.tot5_list = list()
        self.cor_list = list()
        self.tot_list = list()
        self.default_metric_name = default_metric_name
        if default_metric_name is None:
            self.default_metric_name = "R_5"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, pred, label):
        batch_size = len(label)
        for i in range(batch_size):
            cor1, tot1 = self.compute_acc(pred[i], label[i], self.topk[0])
            cor2, tot2 = self.compute_acc(pred[i], label[i], self.topk[1])
            cor5, tot5 = self.compute_acc(pred[i], label[i], self.topk[2])
            cor, tot = self.compute_map(pred[i], label[i])
            self.cor1_list.append(cor1)
            self.tot1_list.append(tot1)
            self.cor2_list.append(cor2)
            self.tot2_list.append(tot2)
            self.cor5_list.append(cor5)
            self.tot5_list.append(tot5)
            self.cor_list.append(cor)
            self.tot_list.append(tot)

    def get_metric(self, reset=True):
        evaluate_result = {}
        R_1 = sum(self.cor1_list) / sum(self.tot1_list)
        R_2 = sum(self.cor2_list) / sum(self.tot2_list)
        R_5 = sum(self.cor5_list) / sum(self.tot5_list)
        R = sum(self.cor_list) / sum(self.tot_list)
        evaluate_result = {"R_1": R_1,
                           "R_2": R_2,
                           "R_5": R_5,
                           "R": R,
                           }

        if reset:
            self.cor1_list = list()
            self.tot1_list = list()
            self.cor2_list = list()
            self.tot2_list = list()
            self.cor5_list = list()
            self.tot5_list = list()
            self.cor_list = list()
            self.tot_list = list()
        return evaluate_result

    def compute_acc(self, sorted_idx, labels, k=5):
        sorted_idx = sorted_idx.unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0)
        idx = sorted_idx[:, :k]
        labels = labels.unsqueeze(-1).expand_as(idx)
        cor_ts = torch.eq(idx, labels).any(dim=-1)
        cor = cor_ts.sum().item()
        total = cor_ts.numel()
        return cor, total

    def compute_map(self, sorted_idx, labels):
        sorted_idx = sorted_idx.unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0)
        l = labels.unsqueeze(-1)
        w = sorted_idx == l
        _, idx = w.nonzero(as_tuple=True)
        s = 1. / (idx + 1)
        return torch.sum(s).item(), labels.size(0)
