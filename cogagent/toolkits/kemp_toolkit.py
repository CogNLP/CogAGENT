import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.models.base_kemp_model import KempModel
from cogagent.data.datable import DataTable
# from test.kemp_base import load_params
from cogagent.data.processors.kemp_processors.kemp_processor import convert_to_tensor, preprocess
from collections import defaultdict
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent.data.processors.kemp_processors.kemp_dataloader import Dataset
import argparse
import torch
import json
from cogagent import load_model


def load_params():
    if (os.cpu_count() > 8):
        USE_CUDA = True
    else:
        USE_CUDA = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="data/kemp_dataset_preproc.json",
                        help='processed EmpatheticDialogue dataset')
    parser.add_argument("--save_path", type=str, default="save/test/", help='path to save the training files')
    parser.add_argument("--resume_path", type=str, default="result/", help='path to save the checkpoint file')
    parser.add_argument("--tokenizer_dir", type=str, default="data/", help='path to tokenization file')
    parser.add_argument("--emb_file", type=str, default='', help='path to glove embedding file')

    ## training
    parser.add_argument("--model", type=str, default="seq2seq", help='model name, [KEMP, wo_ECE, wo_EDD]')
    parser.add_argument("--use_cuda", type=bool, default=True, help='gpu is available or not')
    parser.add_argument("--cuda", action="store_true", help='use gpu or not')
    parser.add_argument('--device_id', dest='device_id', type=str, default="8", help='gpu device id')
    parser.add_argument('--eps', type=float, default=1e-9, help='arg in NoamOpt')
    parser.add_argument('--epochs', type=int, default=10000, help='training iterations')
    parser.add_argument('--check_iter', type=int, default=2000, help='validation iterations')
    parser.add_argument("--noam", action="store_true", help='NoamOpt')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2, help='dropout')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument("--plm", action="store_true", help='use pretraining model or not')
    parser.add_argument("--use_oov_emb", action="store_true", help='')
    parser.add_argument("--pretrain_emb", action="store_true", help='use pretrained embedding (glove) or not')
    parser.add_argument("--projection", action="store_true", help='projection')
    parser.add_argument("--weight_sharing", action="store_true",
                        help='sharing params between input embedding and output proj')
    # parser.add_argument("--label_smoothing", action="store_true", help='label smoothing loss')
    parser.add_argument("--universal", action="store_true", help='universal transformer')
    parser.add_argument("--act", action="store_true", help='arg in universal transformer, adaptive computation time')
    parser.add_argument("--act_loss_weight", type=float, default=0.001, help='arg in universal transformer')
    parser.add_argument("--specify_model", action="store_true", help='arg for resuming training')

    ## testing
    parser.add_argument("--test", action="store_true", help='true for inference, false for training')
    parser.add_argument("--train_then_test", action="store_true", help='test model if the training finishes')
    parser.add_argument("--beam_search", action="store_true", help='beam decoding')
    parser.add_argument("--beam_size", type=int, default=5, help='beam size')
    parser.add_argument("--topk", type=int, default=0, help='topk sampling')

    ## transformer
    parser.add_argument("--hidden_dim", type=int, default=100, help='hidden size')
    parser.add_argument("--emb_dim", type=int, default=100, help='embedding dimension')
    parser.add_argument("--hop", type=int, default=6, help='number of transformer layers')
    parser.add_argument("--heads", type=int, default=1, help='number of attention heads')
    parser.add_argument("--depth", type=int, default=40,
                        help='size of last dimension of keys/values. Must be divisible by number of heads')
    parser.add_argument("--filter", type=int, default=50, help='hidden size of the middle layer in FFN.')
    parser.add_argument("--project", action="store_true",
                        help='project the input of decoder from embedding dimension to hidden dimension')
    parser.add_argument("--concept_num", type=int, default=3,
                        help='the maximum number of external concepts injection for a word.')
    parser.add_argument("--total_concept_num", type=int, default=10,
                        help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--max_seq_length", type=int, default=1000,
                        help='max sequence length (required for timing signal)')
    parser.add_argument("--pointer_gen", action="store_true", help='copy mechanism')
    parser.add_argument("--attn_loss", action="store_true", help="emotional attention loss")
    parser.add_argument("--emotion_feature", action="store_true", help="emotional feature")

    args = parser.parse_args()
    # print_opts(args)

    args.emb_file = args.emb_file or "data/glove.6B.{}d.txt".format(str(args.emb_dim))
    if (not args.test):
        args.save_path_dataset = args.save_path

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
    #                     datefmt='%m-%d %H:%M')  # ,filename='save/logs/{}.log'.format(str(name)))
    args.collect_stats = False

    args.UNK_idx = 0
    args.PAD_idx = 1
    args.EOS_idx = 2
    args.SOS_idx = 3
    args.USR_idx = 4  # speak state
    args.SYS_idx = 5  # listener state
    args.KG_idx = 6  # concept state
    args.CLS_idx = 7
    args.SEP_idx = 8
    args.device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
    args.USE_CUDA = USE_CUDA
    return args


args = load_params()
os.environ["CUDA_VISOBLE_DEVICES"] = args.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device_id))


class KempToolkit(BaseToolkit):
    def __init__(self, bert_model, model_path, vocabulary_path, device):
        super(KempToolkit, self).__init__(bert_model, model_path, None, device)
        # self.plm = PlmAutoModel(bert_model)
        self.args = load_params()
        with open(vocabulary_path, "rb") as f:
            self.vocab = json.load(f)
        # word2index, word2count, index2word, n_words = self.vocab
        # self.vocabulary = vocab
        self.model = KempModel(args=self.args, vocab=self.vocab, decoder_number=32)
        load_model(self.model, self.model_path)
        self.model.to(self.device)

    def run(self, sentence):
        sentence = re.split(r'\s+', sentence)
        preprocessed_sen = preprocess(sentence, concept_num=3)
        datable = DataTable()
        datable('context', [preprocessed_sen['context']])
        datable('concepts', preprocessed_sen['concepts'])
        datable('vads', preprocessed_sen['vads'])
        datable('vad', preprocessed_sen['vad'])

        datable('sample_concepts', preprocessed_sen['sample_concepts'])
        datable('target_vad', preprocessed_sen['target_vad'])
        datable('target_vads', preprocessed_sen['target_vads'])
        datable('target', preprocessed_sen['target'])
        datable('emotion', preprocessed_sen['emotion'])
        datable('situation', preprocessed_sen['situation'])
        # 数据预处理：得到 加入概念和VAD后，包含句子相关信息的字典（即训练集）
        item = Dataset(datable, self.vocab[0], self.args)
        item = item.__getitem__(0)
        input_dict = convert_to_tensor(self.args, item)
        # 输入数据处理：得到模型输入的数据格式
        move_dict_value_to_device(input_dict, self.device)
        label = self.model.predict_label(input_dict)
        # label = self.vocabulary.id2label(label_id.clone().detach().cpu().item())
        return label


if __name__ == '__main__':
    toolkit = KempToolkit(bert_model=None,
                          model_path='/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/experimental_result/simple_test--2023-01-30--17-28-17.80/model/checkpoint-50000/models.pt',
                          vocabulary_path='/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data/vocab.json',
                          device=torch.device("cuda:8"),
                          )
    sentence = "I am very happy today."
    label = toolkit.run(sentence)
    print("label:", label[0])
