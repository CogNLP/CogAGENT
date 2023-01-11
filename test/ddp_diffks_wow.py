# CUDA_VISIBLE_DEVICES="6,7,8,9"  python -m torch.distributed.launch --nproc_per_node 4 test_base_sentence_pair_classification.py

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn

from cogagent import *
from cogagent.core.evaluator import Evaluator
from cogagent.models.diffks_model import DiffKSModel
from cogagent.core.metric.base_kgc_metric import BaseKGCMetric

def main(local_rank):
    device,output_path = init_cogagent(
        output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result",
        folder_tag="debug_ddp_diffks_wow_5e-4",
        rank=local_rank,
        seed=1 + local_rank,
    )
    # if local_rank != 0:
    #     dist.barrier()
    # reader = WoWReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/raw_data")
    # train_data, dev_data, test_data = reader.read_all()
    # vocab = reader.read_vocab()
    # if local_rank == 0:
    #     dist.barrier()
    #
    # processor = WoWForDiffksProcessor(max_token_len=512, vocab=vocab, debug=False)
    # train_dataset = processor.process_train(train_data)
    # dev_dataset = processor.process_dev(dev_data)
    # test_dataset = processor.process_test(test_data)
    cache_file = "/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/processor_datas.pkl"
    train_dataset,dev_dataset,test_dataset,vocab = load_pickle(cache_file)
    dist.barrier()

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)
    test_sampler = DistributedSampler(test_dataset)

    model = DiffKSModel(glove_path='/data/hongbang/CogAGENT/datapath/pretrained_models/glove', vocab=vocab)
    metric = BaseKGCMetric(default_metric_name="bleu-4", vocab=vocab)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    dist.barrier()

    trainer = Trainer(model,
                      train_dataset,
                      dev_data=dev_dataset,
                      n_epochs=40,
                      batch_size=2,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=None,
                      metrics=metric,
                      train_sampler=train_sampler,
                      dev_sampler=dev_sampler,
                      drop_last=False,
                      gradient_accumulation_steps=1,
                      num_workers=5,
                      print_every=None,
                      scheduler_steps=None,
                      validate_steps=1000,      # validation setting
                      save_by_metric="bleu-4",
                      save_steps=None,
                      output_path=output_path,
                      grad_norm=1,
                      use_tqdm=True,
                      device=device,
                      callbacks=None,
                      metric_key=None,
                      fp16=False,
                      fp16_opt_level='O1',
                      rank=local_rank,
                      )
    trainer.train()
    print("end")




def spmd_main(local_rank):
    dist.init_process_group(backend="nccl")
    main(local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    spmd_main(local_rank)