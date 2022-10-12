import torch.nn as nn
# import torch.optim as optim
import torch
import torch.nn as nn
import logging

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import get_linear_schedule_with_warmup, AdamW

# 解决 ModuleNotFoundError: No module named 'cogagent' 问题
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogagent.models.dst_sumbt_multiwoz import DstSumbtModel
# from cogagent.models.dst_sumbt_multiwoz import *
# 导入配置文件
from cogagent.data.processors.multiwoz_processors.sumbt_config import *

# 导入 multiwoz_slot_trans
from cogagent.data.processors.multiwoz_processors.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

# 导入state
from cogagent.data.processors.multiwoz_processors.state import default_state

# 导入DSTMetric
from cogagent.core.metric.dst_metric import DSTMetric

# 主要的打印输出代码
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.enabled = False

# from cogagent import *
from cogagent.data import *
from cogagent.core import *
from cogagent.models import *
from cogagent.toolkits import *
from cogagent.utils import *
# from cogagent.data import datable
from cogagent.data.datable import DataTable


# 1、 整个模块的初始化
device, output_path = init_cogagent(  # device(type='cuda', index=0)   '/home/nlp/code/nlp/CogAGENT-main/datapath/text_classification/SST_2/experimental_result/simple_test--2022-09-06--09-14-14.20'
    # 支持多GPU训练
    device_id=2,
    # 模型、输出结果 存在哪个路径上
    output_path="/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/experimental_result",
    folder_tag="dst_test",
)


# 2、数据集的读取，读到内存中， 并没有任何处理  #reader 是 Sst2Reader 对象
reader = Dst2Reader(raw_data_path="/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/raw_data")
raw_data_path = '/home/nlp/code/nlp/CogAGENT-main/datapath/dst/multiwoz/raw_data'
second_data = reader._read(raw_data_path)  # <cogagent.data.datable.DataTable object at 0x7f6af3011fd0>
train_data, test_data, dev_data = reader.read_all(second_data)  # 在这返回的是DataTable
# vocab = reader.read_vocab()   # 实际上读取的是标签 返回的是字典，键是label_vocab【是Vocablary的对象】，值是Vocabulary的对象,  <cogagent.utils.vocab_utils.Vocabulary object at 0x7fe73d990150>

# 3、处理数据，处理成数据需要的格式
processor = Multiwoz2Processor(plm="bert-base-uncased")
embeddings = processor.process_embedding(second_data) # 返回slot-value domain-slot-type embeddings

dst_sumbt = DstSumbtModel(args, second_data['num_labels'][0], device)



if args.fp16:
    dst_sumbt.half()
dst_sumbt.to(device)

# self.args = args
state = default_state()
param_restored = False
USE_CUDA = torch.cuda.is_available() # True
# N_GPU = torch.cuda.device_count() if USE_CUDA else 1 # 3
N_GPU = 1
if USE_CUDA and N_GPU == 1:
    dst_sumbt.initialize_slot_value_lookup(embeddings['label_token_ids'][0], embeddings['slot_token_ids'][0])
elif USE_CUDA and N_GPU > 1:
    # self.sumbt_model.module.initialize_slot_value_lookup(self.label_token_ids, self.slot_token_ids)
    dst_sumbt.initialize_slot_value_lookup(embeddings['label_token_ids'][0], embeddings['slot_token_ids'][0])
det_dic = {}
for domain, dic in REF_USR_DA.items():
    for key, value in dic.items():
        assert '-' not in key
        det_dic[key.lower()] = key + '-' + domain
        det_dic[value.lower()] = key + '-' + domain

cached_res = {}
datable_train = DataTable()
datable_dev = DataTable()

## Training utterances
all_input_ids, all_input_len, all_label_ids, train_dataset = processor._process(
            train_data['train_examples'][0], 
            second_data['label'], 
            args.max_seq_length, 
            args.max_turn_length
            )
 # all_input_ids torch.Size([8434, 22, 64]);all_input_len = torch.Size([8434, 22, 2]); all_label_ids = torch.Size([8434, 22, 35])
num_train_batches = all_input_ids.size(0) # 8434
num_train_steps = int(num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

logger.info("***** training *****")
logger.info("  Num examples = %d", len(train_data['train_examples']))
logger.info("  Batch size = %d", args.train_batch_size)
logger.info("  Num steps = %d", num_train_steps)


# all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(device), all_label_ids.to(device)

train_data_ = TensorDataset(all_input_ids, all_input_len, all_label_ids)
# train_data_ = TensorDataset(datable_train)
train_sampler = RandomSampler(train_data_)
train_dataloader = DataLoader(train_data_, sampler=train_sampler, batch_size=args.train_batch_size)
# train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

all_input_ids_dev, all_input_len_dev, all_label_ids_dev, dev_dataset = processor._process(
                dev_data['dev_examples'][0], 
                second_data['label'], 
                args.max_seq_length, 
                args.max_turn_length
                )
# all_input_ids_dev=torch.Size([999, 17, 64]); all_input_len_dev=torch.Size([999, 17, 2]);all_label_ids_dev=torch.Size([999, 17, 35])
logger.info("***** validation *****")
logger.info("  Num examples = %d", len(dev_data['dev_examples']))
logger.info("  Batch size = %d", args.dev_batch_size)

all_input_ids_dev, all_input_len_dev, all_label_ids_dev = all_input_ids_dev.to(device), all_input_len_dev.to(device), all_label_ids_dev.to(device)

dev_data_ = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev)
dev_sampler = SequentialSampler(dev_data_)
dev_dataloader = DataLoader(dev_data_, sampler=dev_sampler, batch_size=args.dev_batch_size)

# dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size)

logger.info("Loaded data!")


# 设置优化器， 学习率 和 衰减率  model 是 dst_sumbt
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

if not USE_CUDA or N_GPU == 1:
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(dst_sumbt)
else:
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(dst_sumbt.module)

t_total = num_train_steps # 42170
scheduler = None

if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
    if args.fp16_loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.fp16_loss_scale)

else:
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion*t_total, num_training_steps=t_total)
logger.info(optimizer)

### Classifier
loss = nn.CrossEntropyLoss(ignore_index=-1)
# metric = BaseClassificationMetric(mode="multi")
metric = DSTMetric("multi", second_data['target_slot'][0], second_data['num_labels'][0])

model = dst_sumbt

# 训练器
trainer = Trainer(model,                      # 模型
                  train_dataset,  
                #   train_data_, 
                  dev_data=dev_dataset,
                #   dev_data = dev_data_,
                  n_epochs=args.num_train_epochs,
                  batch_size=args.train_batch_size,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=10,           # 每多少步验证一下指标
                  save_steps=None,              # 多少步存一下
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")


from tensorboardX.writer import SummaryWriter
from tqdm._tqdm import trange, tqdm
def tt(sumbt_model, train_dataloader, dev_dataloader, second_data):
        global_step = 0
        last_update = None
        best_loss = None
        model = sumbt_model
        if not args.do_not_use_tensorboard:
            summary_writer = None
        else:
            summary_writer = SummaryWriter("./tensorboard_summary/logs_1214/")

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  # args.num_train_epoches = 10
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader)):  # step = 0, batch = 
                batch = tuple(t.to('cuda:2') for t in batch)
                input_ids, input_len, label_ids = batch

                # Forward
                if N_GPU == 1: # loss = tensor(98.6031, device='cuda:0', grad_fn=<AddBackward0>) loss_slot = [2.1217856407165527, 4.917163848876953, 2.9272401332855225, 0.5690963268280029, 0.7756544351577759, 1.5024203062057495, 0.3553546071052551, 3.828152894973755, 2.645986795425415, 2.009312152862549, 2.1946418285369873, 2.1952807903289795, 1.2690727710723877, 4.762704849243164, ...] acc = tensor(0., device='cuda:0') 
                    loss, loss_slot, acc, acc_slot, _ = model(batch) # tensor([0.0769, 0.0000, 0.2308, 1.0000, 0.0000, 0.0769, 1.0000, 0.0000, 0.0000,        0.8462, 0.0000, 0.0000, 0.6923, 0.0000, 1.0000, 0.3846, 0.0000, 0.6154,        0.0000, 0.9231, 0.0000, 1.0000, 0.0000, 0.0000, 0.6154, 0.4615, 0.0000,        0.0000, 0.0769, 0.1538, 0.0000, 0.7692, 0.0000, 0.0000, 0.0769],       device='cuda:0')
                else:
                    loss, _, acc, acc_slot, _ = model(batch)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    if N_GPU == 1:
                        for i, slot in enumerate(second_data["target_slot"][0]):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i],
                                                      global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i], global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0) # 2
                nb_tr_steps += 1                    # 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    if scheduler is not None:
                        for i in range(len(optimizer_grouped_parameters)):
                            torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters[i]['params'], 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1


            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to('cuda:2') for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if N_GPU == 1:
                        loss, loss_slot, acc, acc_slot, _ = model(batch)
                    else:
                        loss, _, acc, acc_slot, _ = model(batch)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0) 

                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                if N_GPU == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn


            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples

            if N_GPU == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                if N_GPU == 1:
                    for i, slot in enumerate(second_data['target_slot'][0]):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'),
                                                  dev_loss_slot[i] / nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i],
                                                  global_step)

            dev_loss = round(dev_loss, 6)

            output_model_file = os.path.join(os.path.join('sumbt_path', 'model_ouput'), "pytorch_model.bin")

            if last_update is None or dev_loss < best_loss:

                if not USE_CUDA or N_GPU == 1:
                    torch.save(model.state_dict(), output_model_file)
                else:
                    torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info(
                    "*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d ***" % (
                        last_update, best_loss, best_acc, global_step))
            else:
                logger.info(
                    "*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f, global_step=%d  ***" % (
                        epoch, dev_loss, dev_acc, global_step))

            if last_update + args.patience <= epoch:
                break


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

# if __name__ == '__main__':
    # tt(model, train_dataloader, dev_dataloader, second_data)