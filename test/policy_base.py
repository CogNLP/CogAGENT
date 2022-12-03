import argparse
from ctypes import util
import json
import torch
import torch.nn as nn
import torch.optim as optim
# 解决 ModuleNotFoundError: No module named 'cogagent' 问题
# from cogagent.data.readers.jointbert_reader import JointbertReader

import sys,os

# from CogAGENT.cogagent.core.metric.base_policy_metric import BasePoliyMetric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogagent.data.readers.policy_reader import PolicyReader
from cogagent.models.base_policy_model import PolicyModel
# from cogagent.core.metric.base_jointbert_metric import BaseJointbertMetric
# from cogagent.data.readers.policy_reader import PolicyReader
from cogagent.utils.policy_util import str2bool
from cogagent.data.processors.policy_processors.policy_processor import PolicyProcessor
from cogagent.core import *
from cogagent.core.metric.base_policy_metric import BasePoliyMetric
from cogagent.models import *
from cogagent.toolkits import *
from cogagent.data.processors.policy_processors.policy_processor import PolicyProcessor
from cogagent.data import *
from cogagent.data.processors import *
from cogagent.data.processors.policy_processors import *
from cogagent.utils import *

device, output_path = init_cogagent(
    # device_id=7,
    device_id=0,
    output_path="/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/policy/experimental_result",
    folder_tag="simple_test",
)
parser = argparse.ArgumentParser(description='S2S')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--vocab_size', type=int, default=400, metavar='V')
parser.add_argument('--use_attn', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--attention_type', type=str, default='bahdanau')
# parser.add_argument('--use_emb',  type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--emb_size', type=int, default=50)
parser.add_argument('--hid_size_enc', type=int, default=150)
parser.add_argument('--hid_size_dec', type=int, default=150)
parser.add_argument('--hid_size_pol', type=int, default=150)
parser.add_argument('--db_size', type=int, default=30)
parser.add_argument('--bs_size', type=int, default=94)
parser.add_argument('--cell_type', type=str, default='lstm')
parser.add_argument('--depth', type=int, default=1, help='depth of rnn')
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr_rate', type=float, default=0.0001)
parser.add_argument('--lr_decay', type=float, default=0.0)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')
parser.add_argument('--teacher_ratio', type=float, default=1.0, help='probability of using targets for learning')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--no_cuda',  type=str2bool, nargs='?', const=True, default=True)
# parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--train_output', type=str, default='data/train_dials/', help='Training output dir path')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--early_stop_count', type=int, default=2)
# parser.add_argument('--model_dir', type=str, default='model/model/')
parser.add_argument('--load_param', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--epoch_load', type=int, default=0)
parser.add_argument('--mode', type=str, default='train', help='training or testing: test, train, RL')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
parser.add_argument('--original', type=str, default='data/multi-woz/model/model/', help='Original path.')
parser.add_argument('--use_emb', type=str, default='False')
# parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
parser.add_argument('--write_n_best', type=str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')
parser.add_argument('--model_path', type=str, default='model/model/translate.ckpt', help='Path to a specific model checkpoint.')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--model_name', type=str, default='translate.ckpt')
parser.add_argument('--valid_output', type=str, default='model/data/val_dials/', help='Validation Decoding output dir path')
parser.add_argument('--decode_output', type=str, default='model/data/test_dials/', help='Decoding output dir path')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
raw_data_path="/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/policy/raw_data"
reader = PolicyReader(raw_data_path)
train_data, dev_data, test_data = reader.read_all()
delex_dialogues = reader.read_delex()
dbs = reader.read_db()
input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = reader.read_vocab()

processor = PolicyProcessor(input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
train_dataset = processor.process_train(train_data, data_key = 'train')
dev_dataset = processor.process_dev(dev_data, data_key = 'val')
test_dataset = processor.process_test(test_data, data_key = 'test')

model = PolicyModel(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index,dev_data)
metric = BasePoliyMetric(dbs = dbs, delex_dialogues = delex_dialogues)

loss = nn.NLLLoss(ignore_index=3, reduction='mean')

optimizer = optim.Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad,model.parameters()), weight_decay=args.l2_norm)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=2,
                  batch_size=1,
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
                  validate_steps=5,
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("end")
