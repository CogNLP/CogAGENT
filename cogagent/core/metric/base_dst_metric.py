from cogagent.core.metric.base_metric import BaseMetric
import torch
n_gpu = 1

class DSTMetric(BaseMetric):
    def __init__(self, mode, target_slot, num_labels, default_metric_name=None):
        super().__init__()
        if mode not in ["binary", "multi"]:
            raise ValueError("Please choose mode in binary or multi")
        self.mode = mode
        self.label_list = list()
        self.pre_list = list()
        self.num_valid_turn = list()
        self.default_metric_name = default_metric_name # None
        if default_metric_name is None:
            self.default_metric_name = "F1" if mode == "binary" else "macro_F1"  # default_metric_name = F1
        else:
            self.default_metric_name = default_metric_name
        
        self.default_metric_name = 'Acc'
        self.num_slots = len(num_labels) 
        self.target_slot = target_slot
        

    def evaluate(self, pred, label):
        # self.label_list = self.label_list.append(label) # 问题一
        # self.pre_list = self.pre_list.append(pred)
        # self.num_valid_turn = self.num_valid_turn.append(torch.sum(label[:, :, 0].view(-1) > -1, 0).item())
        self.label_list.append(label) # 问题一
        self.pre_list.append(pred)
        self.num_valid_turn.append(torch.sum(label[:, :, 0].view(-1) > -1, 0).item())

        # num_valid_turn = torch.sum(batch[2][:, :, 0].view(-1) > -1, 0).item()
        # dev_loss += loss.item() * num_valid_turn
        # dev_acc += acc * num_valid_turn

        # self.label_list = self.label_list + label.cpu().tolist()
        # self.pre_list = self.pre_list + pred.cpu().tolist()
        # target_slot = list(ontology.keys()) # ['attraction-area', 'attraction-name', 'attraction-type', 'bus-day', 'bus-departure', 'bus-destination', 'bus-leaveAt', 'hospital-department', 'hotel-area', 'hotel-book day', 'hotel-book people', 'hotel-book stay', 'hotel-internet', 'hotel-name', ...]
         ### Measure

        pass
    def get_metric(self, reset=True):  #　变量没用上
        evaluate_result = {}
         # 新加入
        dev_loss = 0
        dev_acc = 0
        dev_loss_slot, dev_acc_slot = None, None
        nb_dev_examples, nb_dev_steps = 0, 0

        if self.target_slot is None:
            self.target_slot = list(range(0, self.num_slots))
        slot_dim = len(self.target_slot)
        for i in range(len(self.label_list)):

            accuracy = (self.label_list[i] == self.pre_list[i]).view(-1, slot_dim)
            acc_slot = torch.sum(accuracy, 0).float() / torch.sum(self.label_list[i].view(-1, slot_dim) > -1, 0).float()
            acc = sum(torch.sum(accuracy, 1) / slot_dim).float() / torch.sum(self.label_list[i][:, :, 0].view(-1) > -1, 0).float()  # joint accuracy
            dev_acc += acc.item() * self.num_valid_turn[i]
            nb_dev_examples += self.num_valid_turn[i]

        # dev_loss = dev_loss / nb_dev_examples
        dev_acc = dev_acc / nb_dev_examples
        evaluate_result =  {
                                "Acc": dev_acc,
                                # "Acc_slot": acc_slot,
                            }
        if n_gpu == 1:
            # return loss, loss_slot, acc, acc_slot, pred_slot
            evaluate_result =  {
                                "Acc": dev_acc,
                                # "Acc_slot": acc_slot,
                            }
        else:
            evaluate_result =  {
                                "Acc": dev_acc.unsqueeze(0),
                                # "Acc_slot": acc_slot.unsqueeze(0),
                               }
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result