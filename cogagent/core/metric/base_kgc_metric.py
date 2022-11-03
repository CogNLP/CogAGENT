from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BaseKGCMetric(BaseMetric):
    def __init__(self, default_metric_name=None):
        super().__init__()

        self.hyps_list = list()
        self.refs_list = list()
        self.default_metric_name = default_metric_name
        self.ignore_smoothing_error = False
        if default_metric_name is None:
            self.default_metric_name = "bleu-4"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, hyps, refs,sent_num):
        refs = refs.cpu().numpy()
        hyps = hyps.cpu().numpy().transpose(2, 0, 1)
        for i,turn_length in enumerate(sent_num):
            gen_session = hyps[i]
            ref_session = refs[i]
            for j in range(turn_length):
                self.hyps_list.append(gen_session[j].tolist())
                self.refs_list.append([ref_session[j].tolist()])


    def get_metric(self, reset=True):
        evaluate_result = {}
        for i in range(1, 5):
            try:
                weights = [1. / i] * i + [0.] * (4 - i)
                evaluate_result.update(
                    {"bleu-%d" % i: 100 * corpus_bleu(self.refs_list, self.hyps_list, weights,
                                                      smoothing_function=SmoothingFunction().method3)})
            except ZeroDivisionError as _:
                if not self.ignore_smoothing_error:
                    raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                    usually caused when there is only one sample and the sample length is 1.")
                evaluate_result.update({"bleu-%d" % i: 0})
        return evaluate_result
        # if self.mode == "binary":
        #     P = precision_score(self.label_list, self.pre_list, average="binary")
        #     R = recall_score(self.label_list, self.pre_list, average="binary")
        #     F1 = f1_score(self.label_list, self.pre_list, average="binary")
        #     Acc = accuracy_score(self.label_list, self.pre_list)
        #     evaluate_result = {"P": P,
        #                        "R": R,
        #                        "F1": F1,
        #                        "Acc": Acc,
        #                        }
        # if self.mode == "multi":
        #     micro_P = precision_score(self.label_list, self.pre_list, average="micro")
        #     micro_R = recall_score(self.label_list, self.pre_list, average="micro")
        #     micro_F1 = f1_score(self.label_list, self.pre_list, average="micro")
        #     macro_P = precision_score(self.label_list, self.pre_list, average="macro")
        #     macro_R = recall_score(self.label_list, self.pre_list, average="macro")
        #     macro_F1 = f1_score(self.label_list, self.pre_list, average="macro")
        #     Acc = accuracy_score(self.label_list, self.pre_list)
        #     evaluate_result = {"micro_P": micro_P,
        #                        "micro_R": micro_R,
        #                        "micro_F1": micro_F1,
        #                        "macro_P": macro_P,
        #                        "macro_R": macro_R,
        #                        "macro_F1": macro_F1,
        #                        "Acc": Acc,
        #                        }
        # if reset:
        #     self.label_list = list()
        #     self.pre_list = list()
        # return evaluate_result