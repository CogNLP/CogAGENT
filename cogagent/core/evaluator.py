import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from cogagent.utils.io_utils import load_model
from cogagent.utils.log_utils import logger
from cogagent.utils.train_utils import move_dict_value_to_device

class Evaluator:
    def __init__(
            self,
            model,
            dev_data,
            metrics,
            checkpoint_path="",
            sampler=None,
            collate_fn=None,
            drop_last=False,
            file_name="models.pt",
            batch_size=32,
            device="cpu",
            user_tqdm=True,
    ):
        """
        在指定数据据上验证模型指标
        :param model: 待验证模型
        :param dev_data: 验证数据集
        :param metrics: 验证指标
        :param checkpoint_path: 模型参数文件所在目录
        :param sampler: 验证数据集对应的采样器
        :param collate_fn: 拼接为batch的函数
        :param drop_last: 是否丢掉最后一个数据
        :param file_name: 模型参数文件的名称，默认为models.pt
        :param batch_size: 验证时的batch_size
        :param device: 模型移动到哪个设备上
        :param user_tqdm: 是否使用tqdm进度条
        """


        self.model = model
        self.checkpoint_path = checkpoint_path
        self.dev_data = dev_data
        self.metrics = metrics
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.file_name = file_name
        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = user_tqdm

        self.dev_dataloader = DataLoader(dataset=self.dev_data, batch_size=self.batch_size,
                                         sampler=self.sampler, drop_last=self.drop_last,
                                         collate_fn=self.collate_fn)

        model_file = os.path.abspath(os.path.join(self.checkpoint_path,file_name))
        if os.path.isfile(model_file):
            self.model = load_model(self.model,model_file)
        else:
            print("Pretrained model file {} does not exist!".format(model_file))

        self.model.to(self.device)


    def evaluate(self):
        logger.info("Start Evaluating...")
        self.model.eval()
        if self.use_tqdm:
            progress = enumerate(tqdm(self.dev_dataloader, desc="Evaluating", leave=False), 1)
        else:
            progress = enumerate(self.dev_dataloader, 1)
        with torch.no_grad():
            for step, batch in progress:
                move_dict_value_to_device(batch, device=self.device)
                self.model.evaluate(batch, self.metrics)
        evaluate_result = self.metrics.get_metric()
        logger.info("Evaluate result = %s", str(evaluate_result))
