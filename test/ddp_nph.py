# CUDA_VISIBLE_DEVICES="1,2,4,7"  python -m torch.distributed.launch --nproc_per_node 4 ddp_nph.py

import torch.optim
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from cogagent.data.readers.open_dialkg_reader import OpenDialKGReader
from cogagent.data.processors.open_dialkg_processors.open_dialkg_for_nph_processor import OpenDialKGForNPHProcessor
from cogagent import init_cogagent,Trainer,BaseClassificationMetric
from cogagent.models.neural_path_hunter_model import MaskRefineModel
from cogagent.core.metric.nph_metric import BaseNPHMetric

def main(local_rank):
    device,output_path = init_cogagent(
        output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/experimental_result",
        folder_tag="debug_ddp_nph",
        rank=local_rank,
        seed=1 + local_rank,
    )
    if local_rank != 0:
        dist.barrier()
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data", debug=True)
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    if local_rank == 0:
        dist.barrier()

    plm_name = 'gpt2'
    mlm_name = 'roberta-large'

    processor = OpenDialKGForNPHProcessor(vocab=vocab, plm=plm_name, mlm=mlm_name, debug=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)

    dist.barrier()

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)

    model = MaskRefineModel(plm_name=plm_name, mlm_name=mlm_name, vocab=vocab)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=6.25e-5, eps=1e-8)
    metric = BaseNPHMetric()

    dist.barrier()
    trainer = Trainer(model,
                      train_dataset,
                      dev_data=dev_dataset,
                      n_epochs=10,
                      batch_size=1,
                      loss=None,
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
                      validate_steps=1200,
                      save_steps=None,
                      output_path=output_path,
                      grad_norm=1,
                      use_tqdm=True,
                      device=device,
                      fp16=False,
                      fp16_opt_level='O1',
                      collate_fn=processor._collate,
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