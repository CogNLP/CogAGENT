from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch


class BaseNPHMetric(BaseMetric):
    def __init__(self, default_metric_name="hits10"):
        super().__init__()
        self.default_metric_name = default_metric_name
        self.val_outputs = []

    def evaluate(self, val_output):
        self.val_outputs.append(val_output)

    def get_metric(self, reset=True):
        val_loss_mean = torch.stack([x["val_loss"] / x["num_tokens"] for x in self.val_outputs]).mean().detach().cpu()
        # self.log("valid/lm_loss", val_loss_mean, prog_bar=True)
        val_ppl = torch.exp(val_loss_mean)

        metric_result = dict(
            lm_ppl=val_ppl.item()
        )
        for metric in ("nce_loss", "nce_accuracy", "mean_reciprocal_rank", "mean_rank", "hits1", "hits3", "hits10"):
            metric_result[metric] = torch.stack([x[metric] for x in self.val_outputs if metric in x]).mean().detach().cpu().item()
        if reset:
            self.val_outputs = []
        return metric_result
