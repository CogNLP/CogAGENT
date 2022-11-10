import torch
import numpy as np
import random
import json
import logging
from cogagent.data.readers.KE_Blender_reader import KE_BlenderReader
from cogagent.data.processors.KE_Blender_processors.KE_Blender_processors import KE_BlenderProcessor
from cogagent.models.KE_Blender.KE_Blender_model import BlenderbotSmallForConditionalGeneration
import torch.nn as nn
import torch.optim as optim
from cogagent.core.metric.KE_Blender_metric import KE_BlenderMetric

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogagent import *
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BlenderbotConfig,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotSmallConfig,
    # BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

device, output_path = init_cogagent(
    device_id=0,
    output_path="/home/nlp/CogAGENT/datapath/KE_Blender_data/experimental_result",
    folder_tag="simple_test",
)

MODEL_CLASSES = {
    "blender": (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer),
    "blender-large": (BlenderbotConfig, BlenderbotForConditionalGeneration, BlenderbotTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES["blender"]


def word_generate(test_data):
    model.to(device)

    all_outputs = []
    # Batching
    # for batch in [
    #     test_data[i : i + self.args.eval_batch_size] for i in range(0, len(test_data), self.args.eval_batch_size)
    # ]:
    for batch in [
        test_data[i:] for i in range(0, len(test_data),)
    ]:
        input_ids = encoder_tokenizer.batch_encode_plus(
            batch, max_length=512, padding='max_length', truncation=True, return_tensors="pt",
        )["input_ids"]
        input_ids = input_ids.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            num_beams=1,
            max_length=128,
            length_penalty=2.0,
            early_stopping=True,
            repetition_penalty=1.0,
            do_sample=False,
            top_k=10,
            top_p=1.0,
            num_return_sequences=1,
            # temperature=0.7
        )

        all_outputs.extend(outputs.cpu().numpy())

    # if self.args.use_multiprocessed_decoding:
    #     self.model.to("cpu")
    #     with Pool(self.args.process_count) as p:
    #         outputs = list(
    #             tqdm(
    #                 p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize),
    #                 total=len(all_outputs),
    #                 desc="Decoding outputs",
    #                 disable=self.args.silent,
    #             )
    #         )
    #     self._move_model_to_device()
    # else:
    outputs = [
        decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for output_id in all_outputs
    ]

    # if self.args.num_return_sequences > 1:
    #     return [
    #         outputs[i: i + self.args.num_return_sequences]
    #         for i in range(0, len(outputs), self.args.num_return_sequences)
    #     ]
    # else:
    #     return outputs
    return outputs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

scheduler = None

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

reader = KE_BlenderReader(raw_data_path="/home/nlp/CogAGENT/datapath/KE_Blender_data/")
# train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = KE_BlenderProcessor(plm="bert-base-cased", vocab=vocab)
# dev_dataset = processor.process_dev(dev_data['dev_data'])
# test_dataset = processor.process_test(test_data['test_data'])
# train_dataset = processor.process_train(train_data['train_data'])

encoder_tokenizer = tokenizer_class.from_pretrained("/home/nlp/CogAGENT/datapath/KE_Blender_data/",
                                                    additional_special_tokens=['__defi__', '__hype__', '[MASK]'])
decoder_tokenizer = encoder_tokenizer
model = BlenderbotSmallForConditionalGeneration.from_pretrained("/home/nlp/CogAGENT/datapath/KE_Blender_data")

model.resize_token_embeddings(len(encoder_tokenizer))

metric = KE_BlenderMetric()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

dev_text = ['Where do you live in?'] # 生成回复的测试数据
reply = word_generate(dev_text)  # 生成对话回复

trainer = Trainer(model,  # 模型
                  train_dataset,  # 训练数据
                  dev_data=dev_dataset,  # 验证数据/
                  n_epochs=3,
                  batch_size=4,
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
