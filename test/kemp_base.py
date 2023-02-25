from ctypes import util
import json
import torch
import torch.nn as nn
import torch.optim as optim
# 解决 ModuleNotFoundError: No module named 'cogagent' 问题
# from cogagent.data.readers.jointbert_reader import JointbertReader
import argparse
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogagent.data.readers.kemp_reader import KempReader
from cogagent.data.processors.kemp_processors.kemp_processor import KempProcessor
from cogagent.models.base_kemp_model import KempModel
from cogagent.models.kemp_common_layer import NoamOpt, LabelSmoothing
# from cogagent.core.metric.base_kemp_metric import BaseKempMetric
from cogagent.core import *
from cogagent.models import *
from cogagent.toolkits import *
from cogagent.data import *
from cogagent.data.processors import *
from cogagent.utils import *

device, output_path = init_cogagent(
    # device_id=7,
    device_id=6,
    output_path="/data/hongbang/CogAGENT/datapath/kemp/experimental_result",
    folder_tag="simple_test",
)

def load_params():
    if (os.cpu_count() > 8):
        USE_CUDA = True
    else:
        USE_CUDA = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="data/kemp_dataset_preproc.json", help='processed EmpatheticDialogue dataset')
    parser.add_argument("--save_path", type=str, default="save/test/", help='path to save the training files')
    parser.add_argument("--resume_path", type=str, default="result/", help='path to save the checkpoint file')
    parser.add_argument("--tokenizer_dir", type=str, default="data/", help='path to tokenization file')
    parser.add_argument("--emb_file", type=str, default='', help='path to glove embedding file')

    ## training
    parser.add_argument("--model", type=str, default="seq2seq", help='model name, [KEMP, wo_ECE, wo_EDD]')
    parser.add_argument("--use_cuda", type=bool, default=True, help='gpu is available or not')
    parser.add_argument("--cuda", action="store_true", help='use gpu or not')
    parser.add_argument('--device_id', dest='device_id', type=str, default="6", help='gpu device id')
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
    parser.add_argument("--weight_sharing", action="store_true", help='sharing params between input embedding and output proj')
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
    parser.add_argument("--depth", type=int, default=40, help='size of last dimension of keys/values. Must be divisible by number of heads')
    parser.add_argument("--filter", type=int, default=50, help='hidden size of the middle layer in FFN.')
    parser.add_argument("--project", action="store_true", help='project the input of decoder from embedding dimension to hidden dimension')
    parser.add_argument("--concept_num", type=int, default=3, help='the maximum number of external concepts injection for a word.')
    parser.add_argument("--total_concept_num", type=int, default=10, help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--max_seq_length", type=int, default=1000, help='max sequence length (required for timing signal)')
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
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.USE_CUDA = USE_CUDA
    return args

args = load_params()

os.environ["CUDA_VISOBLE_DEVICES"] = args.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device_id))
    
raw_data_path="/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data"
reader = KempReader(raw_data_path)
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
word2index, word2count, index2word, n_words = vocab

processor = KempProcessor(word2index, args)

train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)
program_number = 32
# program_number = len(train_dataset.emo_map)

model = KempModel(args, vocab, decoder_number=program_number)
metric = BaseClassificationMetric(mode="binary")
loss = {"nllloss" : nn.NLLLoss(ignore_index=args.PAD_idx),
        "crossentropyloss" : nn.CrossEntropyLoss(reduction='sum')}
# if args.label_smoothing:
#     loss = LabelSmoothing(size=model.vocab_size, padding_idx=args.PAD_idx, smoothing=0.1)
#     loss_ppl = nn.NLLLoss(ignore_index=args.PAD_idx)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.noam:
    optimizer = NoamOpt(args.hidden_dim, 1, 8000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# optimizer = optim.Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad,model.parameters()), weight_decay=args.l2_norm)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=10,
                  batch_size=16,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=400000000000000,
                  save_steps=10000,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.train_collate_fn,
                  dev_collate_fn=dev_dataset.train_collate_fn,
                  )
trainer.train()
print("end")
