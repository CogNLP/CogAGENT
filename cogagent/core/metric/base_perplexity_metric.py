from cogagent.core.metric.base_metric import BaseMetric
import torch


class BasePerplexityMetric(BaseMetric):

    def __init__(self,
                 default_metric_name=None,
                 display_example=None):
        super().__init__()
        self.label_list = list()
        self.pre_list = {}
        self.pre_list['ntokens'] = []
        self.pre_list['ntokens_2'] = []
        self.pre_list['tokens_loss_1'] = []
        self.pre_list['tokens_loss_2'] = []
        self.pre_list['sents_loss_1'] = []
        self.pre_list['sents_loss_2'] = []
        self.pre_list['example'] = []
        self.default_metric_name = default_metric_name
        self.display_example = display_example

    def evaluate(self, pred, label=[]):
        self.pre_list['ntokens'].extend(pred['ntokens'])
        self.pre_list['ntokens_2'].extend(pred['ntokens_2'])
        self.pre_list['tokens_loss_1'].extend(pred['tokens_loss_1'])
        self.pre_list['tokens_loss_2'].extend(pred['tokens_loss_2'])
        self.pre_list['sents_loss_1'].extend(pred['sents_loss_1'])
        self.pre_list['sents_loss_2'].extend(pred['sents_loss_2'])
        self.pre_list['example'].extend(pred['example'])
        self.label_list.extend(label)

    def get_metric(self, reset=True):
        evaluate_result = {}
        tokens_loss_1_numerator = torch.sum(torch.tensor(self.pre_list['tokens_loss_1']))
        tokens_loss_1_denominator = torch.sum(torch.tensor(self.pre_list['ntokens']))
        tokens_loss_1 = (tokens_loss_1_numerator / tokens_loss_1_denominator).item()

        tokens_loss_2_numerator = torch.sum(torch.tensor(self.pre_list['tokens_loss_2']))
        tokens_loss_2_denominator = torch.sum(torch.tensor(self.pre_list['ntokens_2']))
        tokens_loss_2 = (tokens_loss_2_numerator / tokens_loss_2_denominator).item()

        sents_loss_1 = torch.mean(torch.tensor(self.pre_list['sents_loss_1'])).item()
        sents_loss_2 = torch.mean(torch.tensor(self.pre_list['sents_loss_2'])).item()
        evaluate_result = {'tokens_ppl_1': tokens_loss_1,
                           'tokens_ppl_2': tokens_loss_2,
                           'sents_ppl_1': sents_loss_1,
                           'sents_ppl_2': sents_loss_2,
                           }
        if self.display_example is not None:
            for i in self.display_example:
                print(
                    f"persona: {self.pre_list['example'][i]['persona_token'][:150]}\n"
                    f"query: {self.pre_list['example'][i]['query_token'][:100]}\n"
                    f"gold: {self.pre_list['example'][i]['gold_token'][:100]}\n"
                    f"response from D1: {self.pre_list['example'][i]['generated_token'][:100]}\n"
                    f"response from D2: {self.pre_list['example'][i]['generated_token_2'][:100]}\n")

        if reset:
            self.label_list = list()
            self.pre_list = {}
            self.pre_list['ntokens'] = []
            self.pre_list['ntokens_2'] = []
            self.pre_list['tokens_loss_1'] = []
            self.pre_list['tokens_loss_2'] = []
            self.pre_list['sents_loss_1'] = []
            self.pre_list['sents_loss_2'] = []
            self.pre_list['example'] = []
        return evaluate_result
