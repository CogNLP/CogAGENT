from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
import torch
import logging
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor

# transformers.logging.set_verbosity_error()  # set transformers logging level


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id

class Multiwoz2Processor(BaseProcessor):
    
    def __init__(self, plm, vocab=None):
        super().__init__()
        self.plm = plm
        # self.max_token_len = max_token_length 
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    
    # 得到embeddings数据统一方法
    def _process_embeddings(self, data): # data是当前标签的所有值
        datable = DataTable()
        ## Get slot-value embeddings
        self.label_token_ids, self.label_len = [], []
        for labels in data['label']:
            # token_ids, lens = self.get_label_embedding(labels, args.max_label_length, self.tokenizer, self.device)
            token_ids, lens = self.get_label_embedding(labels, 32, "cuda:2")
            self.label_token_ids.append(token_ids)
            self.label_len.append(lens)
        self.label_map = [{label: i for i, label in enumerate(labels)} for labels in data['label']]
        self.label_map_inv = [{i: label for i, label in enumerate(labels)} for labels in data['label']]
        self.label_list = data['label']
        self.target_slot = data['target_slot'][0]
        ## Get domain-slot-type embeddings
        self.slot_token_ids, self.slot_len = self.get_label_embedding(data['target_slot'][0], 32, "cuda:2")
            # self.get_label_embedding(data['target_slot'], args.max_label_length, self.tokenizer, self.device)
        
        datable('label_token_ids', self.label_token_ids)
        datable('label_len', self.label_len)
        datable('label_map', self.label_map)
        datable('label_map_inv', self.label_map_inv)
        datable('label_list', self.label_list)
        datable('target_slot', self.target_slot)
        datable('slot_token_ids', self.slot_token_ids)
        datable('slot_len', self.slot_len)

        return datable

    # 得到embedding方法
    def process_embedding(self, data):
        return self._process_embeddings(data)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # 处理数据，得到训练数据和测试数据
    def _process(self, examples, label_list, max_seq_length, max_turn_length):
        """Loads a data file into a list of `InputBatch`s."""
        datable = DataTable()
        label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
        slot_dim = len(label_list)

        features = []
        prev_dialogue_idx = None
        all_padding = [0] * max_seq_length
        all_padding_len = [0, 0]

        max_turn = 0
        for (ex_index, example) in enumerate(examples):
            if max_turn < int(example.guid.split('-')[2]):
                max_turn = int(example.guid.split('-')[2])
        max_turn_length = min(max_turn + 1, max_turn_length)
        logger.info("max_turn_length = %d" % max_turn)

        for (ex_index, example) in enumerate(examples):
            tokens_a = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(example.text_a)]
            tokens_b = None
            if example.text_b:
                tokens_b = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(example.text_b)]
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_len = [len(tokens), 0]

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                input_len[1] = len(tokens_b) + 1

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            assert len(input_ids) == max_seq_length

            FLAG_TEST = False
            if example.label is not None:
                label_id = []
                label_info = 'label: '
                for i, label in enumerate(example.label):
                    if label == 'dontcare':
                        label = 'do not care'
                    label_id.append(label_map[i][label])
                    label_info += '%s (id = %d) ' % (label, label_map[i][label])

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % example.guid)
                    logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
                    logger.info("label: " + label_info)
            else:
                FLAG_TEST = True
                label_id = None

            curr_dialogue_idx = example.guid.split('-')[1]
            curr_turn_idx = int(example.guid.split('-')[2])

            if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
                if prev_turn_idx < max_turn_length:
                    features += [InputFeatures(input_ids=all_padding,
                                            input_len=all_padding_len,
                                            label_id=[-1] * slot_dim)] \
                                * (max_turn_length - prev_turn_idx - 1)
                assert len(features) % max_turn_length == 0

            if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                input_len=input_len,
                                label_id=label_id))

            prev_dialogue_idx = curr_dialogue_idx
            prev_turn_idx = curr_turn_idx

        if prev_turn_idx < max_turn_length:
            features += [InputFeatures(input_ids=all_padding,
                                    input_len=all_padding_len,
                                    label_id=[-1] * slot_dim)] \
                        * (max_turn_length - prev_turn_idx - 1)
        assert len(features) % max_turn_length == 0

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # for f in features:
        #     datable("all_input_ids", torch.tensor(f.input_ids, dtype=torch.long).view(-1, max_turn_length, max_seq_length)) 
        
        all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
        # for f in features:
        #     datable('all_input_len', torch.tensor(f.input_len, dtype=torch.long).view(-1, max_turn_length, 2)) 
        
        if not FLAG_TEST:
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            # for f in features:
            #     datable('all_label_ids', torch.tensor(f.label_id, dtype=torch.long).view(-1, max_turn_length, slot_dim))

        # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
        all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
        all_input_len = all_input_len.view(-1, max_turn_length, 2)

        for input in range(len(all_input_ids)):
            datable('all_input_ids', all_input_ids[input])
            datable('all_input_len', all_input_len[input])

        if not FLAG_TEST:
            all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)
            for input in range(len(all_label_ids)):
                datable('all_label_ids', all_label_ids[input])
        else:
            all_label_ids = None
        # datable('all_input_ids', all_input_ids)
        # datable('all_input_len', all_input_len)
        # datable('all_label_ids', all_label_ids)
        return all_input_ids, all_input_len, all_label_ids, DataTableSet(datable)
        # return datable
        
    # 单独处理train方法
    def process_train(self, data):
        return self._process(data)


    # 单独处理test方法
    def process_test(self, data):
        return self._process(data)


    # 单独处理dev的方法
    def process_dev(self, data):
        return self._process(data)

    def get_label_embedding(self, labels, max_seq_length, device):  # ['centre', 'west', 'north', 'south', 'east', 'do not care', 'cambridge', 'none'] 32
        features = []
        for label in labels:  # 'centre'
            label_tokens = ["[CLS]"] + self.tokenizer.tokenize(label) + ["[SEP]"]  # ['[CLS]', 'centre', '[SEP]']
            label_token_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)   # [101, 2803, 102]
            label_len = len(label_token_ids)    # 3 

            label_padding = [0] * (max_seq_length - len(label_token_ids)) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
            label_token_ids += label_padding     # [101, 2803, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
            assert len(label_token_ids) == max_seq_length

            features.append((label_token_ids, label_len))

        all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device) # torch.Size([8, 32])
        all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device) #  tensor([3, 3, 3, 3, 3, 5, 3, 3], device='cuda:0')  torch.Size([8])

        return all_label_token_ids, all_label_len

if __name__ == '__main__':
    pass