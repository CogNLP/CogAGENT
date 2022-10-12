import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import logging
import argparse
import configparser
import time
import sys, os

# 解决 ModuleNotFoundError: No module named 'cogagent' 问题
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from cogagent import *

# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# from transformers import get_linear_schedule_with_warmup, AdamW
from cogagent.data.readers.sclstm_multiwoz_reader import sclstm_multiwoz_reader
from cogagent.data.processors.sclstm_processors.sclstm_processors import Sclstm_Processor
from cogagent.data.readers.sclstm_reader import SimpleDatasetWoz
from cogagent.models.base_sclstm_model import LMDeep
from cogagent.models.base_sclstm_model import get_loss
from cogagent.core.metric.base_sclstm_metric import SclstmMetric

USE_CUDA = torch.cuda.is_available()


def interact(config, args):
    dataset = SimpleDatasetWoz(config)

    # get model hyper-parameters
    n_layer = args.n_layer
    hidden_size = config.getint('MODEL', 'hidden_size')
    dropout = config.getfloat('MODEL', 'dropout')
    lr = args.lr
    beam_size = args.beam_size

    # get feat size
    d_size = dataset.do_size + dataset.da_size + dataset.sv_size  # len of 1-hot feat
    vocab_size = len(dataset.word2index)

    model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=n_layer, dropout=dropout, lr=lr)
    model_path = args.model_path
    print(model_path)
    assert os.path.isfile(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Move models to GPU
    if USE_CUDA:
        model.to(device)

    data_type = args.mode
    print('INTERACT ON: {}'.format(data_type))

    example = {"Attraction-Inform": [["Choice", "many"], ["Area", "centre of town"]],
               "Attraction-Select": [["Type", "church"], ["Type", " swimming"], ["Type", " park"]]}
    for k, v in example.items():
        counter = {}
        for pair in v:
            if pair[0] in counter:
                counter[pair[0]] += 1
            else:
                counter[pair[0]] = 1
            pair.insert(1, str(counter[pair[0]]))

    do_idx, da_idx, sv_idx, featStr = dataset.getFeatIdx(example)
    do_cond = [1 if i in do_idx else 0 for i in range(dataset.do_size)]  # domain condition
    da_cond = [1 if i in da_idx else 0 for i in range(dataset.da_size)]  # dial act condition
    sv_cond = [1 if i in sv_idx else 0 for i in range(dataset.sv_size)]  # slot/value condition
    feats = [do_cond + da_cond + sv_cond]

    feats_var = torch.FloatTensor(feats)
    if USE_CUDA:
        # feats_var = feats_var.cuda()
        feats_var = feats_var.to(device)

    decoded_words = model.generate(dataset, feats_var, beam_size)
    delex = decoded_words[0]  # (beam_size)

    recover = []
    for sen in delex:
        counter = {}
        words = sen.split()
        for word in words:
            if word.startswith('slot-'):
                flag = True
                _, domain, intent, slot_type = word.split('-')
                da = domain.capitalize() + '-' + intent.capitalize()
                if da in example:
                    key = da + '-' + slot_type.capitalize()
                    for pair in example[da]:
                        if (pair[0].lower() == slot_type) and (
                                (key not in counter) or (counter[key] == int(pair[1]) - 1)):
                            sen = sen.replace(word, pair[2], 1)
                            counter[key] = int(pair[1])
                            flag = False
                            break
                if flag:
                    sen = sen.replace(word, '', 1)
        recover.append(sen)

    print('meta', example)
    for i in range(beam_size):
        print(i, delex[i])
        print(i, recover[i])


# 设置优化器， 学习率 和 衰减率  model
def get_optimizer_grouped_parameters(model):
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
    ]
    return optimizer_grouped_parameters


scheduler = None


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():
    parser = argparse.ArgumentParser(description='Train dialogue generator')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--model_path', type=str, default='sclstm.pt', help='saved model path')
    parser.add_argument('--n_layer', type=int, default=1, help='# of layers in LSTM')
    parser.add_argument('--percent', type=float, default=1, help='percentage of training data')
    parser.add_argument('--beam_search', type=str2bool, default=False, help='beam_search')
    parser.add_argument('--attn', type=str2bool, default=True, help='whether to use attention or not')
    parser.add_argument('--beam_size', type=int, default=10, help='number of generated sentences')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')
    parser.add_argument('--user', type=str2bool, default=False, help='use user data')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if args.user:
        config.read('/home/nlp/CogAGENT/cogagent/models/sclstm/multiwoz/config/config_usr.cfg')
    else:
        config.read('/home/nlp/CogAGENT/cogagent/models/sclstm/multiwoz/config/config.cfg')
    config.set('DATA', 'dir', os.path.dirname(os.path.abspath(__file__)))

    return args, config


# args, config = parse()
# if args.mode == 'train' or args.mode == 'adapt':
#     train(config, args)
args, config = parse()

device, output_path = init_cogagent(
    device_id=2,
    output_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_experimental_result",
    folder_tag="simple_test",
)
dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
# output_path = "/home/nlp/CogAGENT/datapath/sclstm_multiwoz_experimental_result"

# 获取model的一些参数
n_layer = args.n_layer  # 1
hidden_size = config.getint('MODEL', 'hidden_size')  # 100
dropout = config.getfloat('MODEL', 'dropout')  # 0.25
lr = args.lr  # 0.0025
beam_size = args.beam_size  # 10
# 获取model的feat size
d_size = dataset.do_size + dataset.da_size + dataset.sv_size  # feat 的one-hot长度
# 词表长度
vocab_size = len(dataset.word2index)  # 1392

model_path = args.model_path

# 模型内部初始化 dataset数据
model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=n_layer, dropout=dropout, lr=lr, use_cuda=USE_CUDA)
model.to(device)

train_data, dev_data, test_data = dataset.read_all()
vocab = dataset.read_vocab(vocab_file_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource/vocab.txt")
word2index = vocab['word2index']
train_data_length = dataset.train_data_length  # 28254
dev_data_length = dataset.valid_data_length  # 3747
test_data_length = dataset.test_data_length  # 3703

processor = Sclstm_Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)

# train_dataset = processor.process_train()
# dev_dataset = processor.process_dev()
# test_dataset = processor.process_test()

train_dataset = processor.process_train(dataset)
dev_dataset = processor.process_dev(dataset)
test_dataset = processor.process_test(dataset)

train_input_var = train_dataset.datable.datas['train_input_var']
train_feats_var = train_dataset.datable.datas['train_feats_var']

# train_label_var = train_dataset.datable.datas['train_label_var']
# train_lengths = train_dataset.datable.datas['train_lengths']

# test_input_var = test_dataset.datable.datas['test_input_var']
# test_feats_var = test_dataset.datable.datas['test_feats_var']

# test_label_va = test_dataset.datable.datas['test_label_var']
# test_lengths = test_dataset.datable.datas['test_lengths']

dev_input_var = dev_dataset.datable.datas['dev_input_var']
dev_feats_var = dev_dataset.datable.datas['dev_feats_var']

# dev_label_var = dev_dataset.datable.datas['dev_label_var']
# dev_lengths = dev_dataset.datable.datas['dev_lengths']

train_data_ = []
dev_data_ = []
test_data_ = []

metric = SclstmMetric()
# loss = nn.CrossEntropyLoss()
# loss = model.get_loss(self, target_label, target_lengths)
loss = get_loss()

optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练器
trainer = Trainer(model,  # 模型
                  train_dataset,  # 训练数据
                  dev_data=dev_dataset,  # 验证数据
                  n_epochs=40,
                  batch_size=64,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=True,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=100,  # 没多少步验证一下指标
                  save_steps=None,  # 多少步存一下
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
