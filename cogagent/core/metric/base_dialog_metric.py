from cogagent.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from rouge import Rouge


class BaseDialogMetric(BaseMetric):

    def __init__(self, valid_mod, default_metric_name=None):
        super().__init__()
        self.valid_mod = valid_mod
        self.rouge = Rouge()
        self.label_list = list()
        self.pre_list = list()
        self.default_metric_name = default_metric_name
        if default_metric_name is None:
            if self.valid_mod == "parallel_valid":
                self.default_metric_name = "macro_F1"
            if self.valid_mod == "serial_valid":
                self.default_metric_name = "rouge-l-f"
        else:
            self.default_metric_name = default_metric_name

    def evaluate(self, pred, label):
        if self.valid_mod == "parallel_valid":
            self.pre_list.extend(pred)
            self.label_list.extend(label)
        if self.valid_mod == "serial_valid":
            pred = [' '.join(list(item)) for item in pred]
            label = [' '.join(list(item)) for item in label]
            self.pre_list.extend(pred)
            self.label_list.extend(label)

    def get_metric(self, reset=True):
        if self.valid_mod == "parallel_valid":
            evaluate_result = {}
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
        if self.valid_mod == "serial_valid":
            evaluate_result = {}
            rouge_1_p_list = []
            rouge_1_r_list = []
            rouge_1_f_list = []
            rouge_2_p_list = []
            rouge_2_r_list = []
            rouge_2_f_list = []
            rouge_l_p_list = []
            rouge_l_r_list = []
            rouge_l_f_list = []
            for (pre, label) in zip(self.pre_list, self.label_list):
                # TODO: when len(fun)=0, we ignore it, what is the standard method
                if len([" ".join(_.split()) for _ in pre.split(".") if len(_) > 0]) > 0:
                    item_result = self.rouge.get_scores(hyps=pre, refs=label)
                    rouge_1_p_list.append(item_result[0]["rouge-1"]["p"])
                    rouge_1_r_list.append(item_result[0]["rouge-1"]["r"])
                    rouge_1_f_list.append(item_result[0]["rouge-1"]["f"])
                    rouge_2_p_list.append(item_result[0]["rouge-2"]["p"])
                    rouge_2_r_list.append(item_result[0]["rouge-2"]["r"])
                    rouge_2_f_list.append(item_result[0]["rouge-2"]["f"])
                    rouge_l_p_list.append(item_result[0]["rouge-l"]["p"])
                    rouge_l_r_list.append(item_result[0]["rouge-l"]["r"])
                    rouge_l_f_list.append(item_result[0]["rouge-l"]["f"])
            sample_num = len(rouge_1_p_list)
            rouge_1_p = sum(rouge_1_p_list) / sample_num
            rouge_1_r = sum(rouge_1_r_list) / sample_num
            rouge_1_f = sum(rouge_1_f_list) / sample_num
            rouge_2_p = sum(rouge_2_p_list) / sample_num
            rouge_2_r = sum(rouge_2_r_list) / sample_num
            rouge_2_f = sum(rouge_2_f_list) / sample_num
            rouge_l_p = sum(rouge_l_p_list) / sample_num
            rouge_l_r = sum(rouge_l_r_list) / sample_num
            rouge_l_f = sum(rouge_l_f_list) / sample_num
            evaluate_result = {"rouge-1-p": rouge_1_p,
                               "rouge-1-r": rouge_1_r,
                               "rouge-1-f": rouge_1_f,
                               "rouge-2-p": rouge_2_p,
                               "rouge-2-r": rouge_2_r,
                               "rouge-2-f": rouge_2_f,
                               "rouge-l-p": rouge_l_p,
                               "rouge-l-r": rouge_l_r,
                               "rouge-l-f": rouge_l_f,
                               }
            if reset:
                self.label_list = list()
                self.pre_list = list()
            return evaluate_result
