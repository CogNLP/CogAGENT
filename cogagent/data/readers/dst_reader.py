import os
import sys
import csv
import zipfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary

# 导入json
import json
import collections

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class Dst2Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.data_file = 'ontology_sumbt.json'
        self.data_path = os.path.join(raw_data_path, self.data_file)
        # self.config = 'all'
        self.label_vocab = Vocabulary()
    
    
    # 转换数据格式 转换成train.tsv test.tsv dev.tsv
    def convert_to_glue_format(self,
            data_dir="/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/raw_data", 
            sumbt_dir='/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz'
        ): # data_dir = '/home/nlp/ConvLab-2/data/multiwoz/raw_data' sumbt_dir = '/home/nlp/ConvLab-2/convlab2/dst/sumbt/multiwoz'

        if not os.path.isdir(os.path.join(sumbt_dir, 'processed_data')):   # tmp_data_dir = processed_data/
            os.mkdir(os.path.join(sumbt_dir, 'processed_data'))

        ### Read ontology file
        fp_ont = open(os.path.join(data_dir, "ontology_sumbt.json"), "r")
        data_ont = json.load(fp_ont)
        ontology = {}
        for domain_slot in data_ont:
            domain, slot = domain_slot.split('-')
            if domain not in ontology:
                ontology[domain] = {}
            ontology[domain][slot] = {}
            for value in data_ont[domain_slot]:
                ontology[domain][slot][value] = 1
        fp_ont.close()

        ### Read woz logs and write to tsv files
        if os.path.exists(os.path.join(sumbt_dir, 'processed_data', "train.tsv")):
            print('data has been processed!')
            return 0

        fp_train = open(os.path.join(sumbt_dir, 'processed_data', "train.tsv"), "w")
        fp_dev = open(os.path.join(sumbt_dir, 'processed_data', "dev.tsv"), "w")
        fp_test = open(os.path.join(sumbt_dir, 'processed_data', "test.tsv"), "w")

        fp_train.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
        fp_dev.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
        fp_test.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')

        for domain in sorted(ontology.keys()):
            for slot in sorted(ontology[domain].keys()):
                fp_train.write(str(domain) + '-' + str(slot) + '\t')
                fp_dev.write(str(domain) + '-' + str(slot) + '\t')
                fp_test.write(str(domain) + '-' + str(slot) + '\t')

        fp_train.write('\n')
        fp_dev.write('\n')
        fp_test.write('\n')

        # fp_data = open(os.path.join(SELF_DATA_DIR, "data.json"), "r")
        # data = json.load(fp_data)

        file_split = ['train', 'val', 'test']
        fp = [fp_train, fp_dev, fp_test]

        for split_type, split_fp in zip(file_split, fp):

            zipfile_name = "{}.json.zip".format(split_type)
            zip_fp = zipfile.ZipFile(os.path.join(data_dir, zipfile_name))
            data = json.loads(str(zip_fp.read(zip_fp.namelist()[0]), 'utf-8'))

            for file_id in data:
                user_utterance = ''
                system_response = ''
                turn_idx = 0
                for idx, turn in enumerate(data[file_id]['log']):
                    if idx % 2 == 0:        # user turn
                        user_utterance = data[file_id]['log'][idx]['text']
                    else:                   # system turn
                        user_utterance = user_utterance.replace('\t', ' ')
                        user_utterance = user_utterance.replace('\n', ' ')
                        user_utterance = user_utterance.replace('  ', ' ')

                        system_response = system_response.replace('\t', ' ')
                        system_response = system_response.replace('\n', ' ')
                        system_response = system_response.replace('  ', ' ')

                        split_fp.write(str(file_id))                   # 0: dialogue ID
                        split_fp.write('\t' + str(turn_idx))           # 1: turn index
                        split_fp.write('\t' + str(user_utterance))     # 2: user utterance
                        split_fp.write('\t' + str(system_response))    # 3: system response

                        belief = {}
                        for domain in data[file_id]['log'][idx]['metadata'].keys():
                            for slot in data[file_id]['log'][idx]['metadata'][domain]['semi'].keys():
                                value = data[file_id]['log'][idx]['metadata'][domain]['semi'][slot].strip()
                                value = value.lower()
                                if value == '' or value == 'not mentioned' or value == 'not given':
                                    value = 'none'

                                if slot == "leaveAt" and domain != "bus":
                                    slot = "leave at"
                                elif slot == "arriveBy" and domain != "bus":
                                    slot = "arrive by"
                                elif slot == "pricerange":
                                    slot = "price range"

                                if value == "doesn't care" or value == "don't care" or value == "dont care" or value == "does not care" or value == 'dontcare':
                                    value = "do not care"
                                elif value == "guesthouse" or value == "guesthouses":
                                    value = "guest house"
                                elif value == "city center" or value == "town centre" or value == "town center" or \
                                        value == "centre of town" or value == "center" or value == "center of town":
                                    value = "centre"
                                elif value == "west part of town":
                                    value = "west"
                                elif value == "mutliple sports":
                                    value = "multiple sports"
                                elif value == "swimmingpool":
                                    value = "swimming pool"
                                elif value == "concerthall":
                                    value = "concert hall"

                                if domain not in ontology:
                                    # print("domain (%s) is not defined" % domain)
                                    continue

                                if slot not in ontology[domain]:
                                    # print("slot (%s) in domain (%s) is not defined" % (slot, domain))   # bus-arriveBy not defined
                                    continue

                                if value not in ontology[domain][slot] and value != 'none':
                                    # print("%s: value (%s) in domain (%s) slot (%s) is not defined in ontology" %
                                    #       (file_id, value, domain, slot))
                                    value = 'none'

                                belief[str(domain) + '-' + str(slot)] = value

                            for slot in data[file_id]['log'][idx]['metadata'][domain]['book'].keys():
                                if slot == 'booked':
                                    continue
                                if domain == 'bus' and slot == 'people':
                                    continue    # not defined in ontology

                                value = data[file_id]['log'][idx]['metadata'][domain]['book'][slot].strip()
                                value = value.lower()

                                if value == '' or value == 'not mentioned' or value == 'not given':
                                    value = 'none'
                                elif value == "doesn't care" or value == "don't care" or value == "dont care" or value == "does not care" or value == 'dontcare':
                                    value = "do not care"

                                if str('book ' + slot) not in ontology[domain]:
                                    # print("book %s is not defined in domain %s" % (slot, domain))
                                    continue

                                if value not in ontology[domain]['book ' + slot] and value != 'none':
                                    # print("%s: value (%s) in domain (%s) slot (book %s) is not defined in ontology" %
                                    #       (file_id, value, domain, slot))
                                    value = 'none'

                                belief[str(domain) + '-book ' + str(slot)] = value

                        for domain in sorted(ontology.keys()):
                            for slot in sorted(ontology[domain].keys()):
                                key = str(domain) + '-' + str(slot)
                                if key in belief:
                                    split_fp.write('\t' + belief[key])
                                else:
                                    split_fp.write('\tnone')

                        split_fp.write('\n')
                        split_fp.flush()

                        system_response = data[file_id]['log'][idx]['text']
                        turn_idx += 1

        fp_train.close()
        fp_dev.close()
        fp_test.close()

    # 读取数据集 将对象和主要的数据存在datable中
    def _read(self, path=None):
        print("Reading multiwoz ontology_sumbt.json")
        datable = DataTable()
        if path == None:
            raise ValueError('Please input file path.')
        with open(self.data_path, mode='r') as f:
            ontology = json.load(f)

        for slot in ontology.keys():
            ontology[slot].append("none")

        # if not config.target_slot == 'all':
        #     slot_idx = {'attraction': '0:1:2', 'bus': '3:4:5:6', 'hospital': '7',
        #                 'hotel': '8:9:10:11:12:13:14:15:16:17', \
        #                 'restaurant': '18:19:20:21:22:23:24', 'taxi': '25:26:27:28', 'train': '29:30:31:32:33:34'}
        #     target_slot = []
        #     for key, value in slot_idx.items():
        #         if key != config.target_slot:
        #             target_slot.append(value)
        #     config.target_slot = ':'.join(target_slot)
        
        # sorting the ontology according to the alphabetic order of the slots
        # 根据插槽的字母顺序对本体进行排序
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        nslots = len(ontology.keys())
        target_slot = list(ontology.keys())
        # if config.target_slot == 'all':
        if 'all' == 'all': 
            self.target_slot_idx = [*range(0, nslots)]
        else:
            # self.target_slot_idx = sorted([int(x) for x in config.target_slot.split(':')])
            pass

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())
        for i, slot in enumerate(self.target_slot):
            if slot == "pricerange":
                self.target_slot[i] = "price range"

        """ 获取此数据集的标签列表。"""
        for slot in self.target_slot:
            datable('label', self.ontology[slot])
        datable('target_slot', self.target_slot)
        datable('target_slot_idx', self.target_slot_idx)
        datable('ontology', self.ontology)
        label_list = [self.ontology[slot] for slot in self.target_slot] 
        num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type
        datable('num_labels', num_labels)
        self.convert_to_glue_format()
        if not os.path.isdir(os.path.join('xxx/项目名称/datapath/dst/multiwoz/', 'model_output/')):
            os.makedirs(os.path.join('xxx/项目名称/datapath/dst/multiwoz/', 'model_output/'))
        # if not os.path.isdir(os.path.join(SUMBT_PATH, args.output_dir)):
        #     os.makedirs(os.path.join(SUMBT_PATH, args.output_dir))
        return datable    
    

    # 读取 .tsv格式的 训练集 测试集 开发集 的通用方法
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # datable = DataTable()
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines

    # 读取开发集
    def _read_dev(self, path, second_data):
        return self._create_examples(
            self._read_tsv(os.path.join(path, "dev.tsv")), 
                            "dev", 
                            second_data,
                            accumulation=False
            )


    # 读取测试集
    def _read_test(self, path, second_data):
        return self._create_examples(
            self._read_tsv(os.path.join(path, "test.tsv")), 
                            "test", 
                            second_data, 
                            accumulation=False
            )

    
    # 读取训练集
    def _read_train(self, path, second_data):
        return self._create_examples(
            self._read_tsv(os.path.join(path, "train.tsv")), 
                            "train", 
                            second_data,
                            accumulation=False
            )

    
    # 读取所有的数据集
    def read_all(self, second_data):
        return self._read_train(
                path='/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/processed_data', second_data=second_data
            ), self._read_test(
                path='/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/processed_data', second_data=second_data
            ), self._read_dev(
                path='/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/processed_data', second_data=second_data
            )

    
    # 
    def read_vocab(self):
        pass


    # 
    def read_addition(self):
        pass


    # 
    def _create_examples(self, lines, set_type, raw_data, accumulation=False):
        """Creates examples for the training and dev sets."""
        datable = DataTable()
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

            label = [line[4 + idx] for idx in raw_data["target_slot_idx"][0]]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        datable(str(f"{set_type}" + "_" + 'examples'), examples)
        return datable

if __name__ == "__main__":
    raw_data_path = '/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/raw_data'
    dst = Dst2Reader(raw_data_path=raw_data_path)
    dst._read_train()

    print("读取数据完成")