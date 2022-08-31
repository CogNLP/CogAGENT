import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from cogagent.utils.io_utils import save_model, load_model
from torch.utils.tensorboard import SummaryWriter
from cogagent.utils.log_utils import logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cogagent.utils.train_utils import reduce_mean, move_dict_value_to_device
import shutil


class Trainer:
    def __init__(
            self,
            model,
            train_data,
            dev_data=None,
            n_epochs=10,
            batch_size=32,
            dev_batch_size=None,
            loss=None,
            optimizer=None,
            scheduler=None,
            metrics=None,
            early_stopping=None,
            train_sampler=None,
            dev_sampler=None,
            drop_last=False,
            gradient_accumulation_steps=1,
            num_workers=0,
            collate_fn=None,
            dev_collate_fn=None,
            print_every=None,
            scheduler_steps=None,
            validate_steps=None,
            output_path=None,
            save_steps=None,
            save_by_metric=None,
            metric_mode='max',
            grad_norm=None,
            use_tqdm=True,
            device=None,
            callbacks=None,
            metric_key=None,
            fp16=False,
            fp16_opt_level='O1',
            checkpoint_path=None,
            rank=-1,
    ):
        """
        训练器构造函数
        :param model:模型
        :param train_data:训练数据
        :param dev_data:验证数据
        :param n_epochs:迭代轮数
        :param batch_size:数据大小
        :param loss:损失函数
        :param optimizer:优化器
        :param scheduler:调整学习率
        :param metrics:评价指标
        :param early_stopping:早停设置
        :param train_sampler:训练集采样器
        :param dev_sampler:验证集采样器
        :param drop_last:丢掉最后一个
        :param gradient_accumulation_steps:梯度累积步数
        :param num_workers:多线程加载数据
        :param print_every:打印步数
        :param save_file:保存模型的文件名
        :param output_path: 实验结果保存的路径
        :param validate_steps:验证步数
        :param save_steps:保存步数
        :param save_by_metric:根据某个metric保存最佳模型
        :param metric_mode: max指标越高越好 min指标越低越好
        :param grad_norm:梯度裁剪
        :param use_tqdm:是否使用tqdm
        :param device:设备
        :param callbacks:回调函数
        :param metric_key:
        """

        self.train_data = train_data
        self.dev_data = dev_data
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.device = device
        self.grad_norm = grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.metrics = metrics
        self.early_stopping = early_stopping
        self.early_stop = False
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.dev_batch_size = dev_batch_size if dev_batch_size is not None else batch_size
        self.use_tqdm = use_tqdm
        self.callbacks = callbacks
        self.train_sampler = train_sampler
        self.dev_sampler = dev_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.dev_collate_fn = dev_collate_fn if dev_collate_fn is not None else collate_fn
        self.drop_last = drop_last
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.checkpoint_path = checkpoint_path
        self.metric_key = metric_key
        self.output_path = output_path
        self.rank = rank
        self.save_by_metric = save_by_metric
        self.metric_mode = metric_mode
        if self.metric_mode not in ["min", "max"]:
            raise ValueError("Metric mode must be min or max but gou {}!".format(self.metric_mode))

        if save_by_metric is not None:
            self.metric_name = self.save_by_metric
        elif hasattr(self.metrics, "default_metric_name"):
            self.metric_name = self.metrics.default_metric_name
        else:
            raise ValueError("Please specified default metric name in the metrics!")
        self.best_metric = {"evaluate_result":None,"global_step":-1}

        if self.rank in [-1, 0]:
            if self.output_path:
                self.writer_path = os.path.join(self.output_path, "tensorboard")
                self.save_path = os.path.join(self.output_path, "model")
                if self.save_by_metric is not None:
                    self.save_best_model_path = os.path.join(self.output_path, "best_model")
                    if not os.path.exists(self.save_best_model_path):
                        os.mkdir(self.save_best_model_path)
                if not os.path.exists(self.writer_path):
                    os.mkdir(self.writer_path)
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
            else:
                self.writer_path = None
                self.save_path = None
        else:
            self.writer_path = None
            self.save_path = None

        if self.rank == -1:
            self.model.to(self.device)
        else:
            self.model = self.model.cuda(self.rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank,
                             find_unused_parameters=False,
                             broadcast_buffers=False)

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, self.optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        self.train_dataloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size,
                                           sampler=self.train_sampler, drop_last=self.drop_last,
                                           collate_fn=self.collate_fn)

        self.batch_count = len(self.train_dataloader)
        if save_steps:
            self.save_steps = save_steps
        else:
            self.save_steps = None

        if print_every:
            self.print_every = print_every
        else:
            self.print_every = self.batch_count

        if scheduler_steps:
            self.scheduler_steps = scheduler_steps
        else:
            self.scheduler_steps = self.batch_count

        if validate_steps:
            self.validate_steps = validate_steps
        else:
            self.validate_steps = self.batch_count

        if self.dev_data:
            self.dev_dataloader = DataLoader(dataset=self.dev_data, batch_size=self.dev_batch_size,
                                             sampler=self.dev_sampler, drop_last=self.drop_last,
                                             collate_fn=self.dev_collate_fn)

        if self.writer_path:
            self.writer = SummaryWriter(self.writer_path)
        else:
            self.writer = SummaryWriter()

    def train(self):
        global_step = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        # Check if continuing training from a checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path) and "checkpoint" in self.checkpoint_path:
            # set global_step to gobal_step of last saved checkpoint from models path
            global_step = int(self.checkpoint_path.split("-")[-1].split("/")[0])
            epochs_trained = epochs_trained + global_step // (
                    len(self.train_dataloader) // self.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(self.train_dataloader) // self.gradient_accumulation_steps)

            logger.info("Continuing training from checkpoint, will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

            if os.path.isfile(os.path.join(self.checkpoint_path, "models.pt")):
                self.model = load_model(self.model, os.path.join(self.checkpoint_path, "models.pt"))
            if os.path.isfile(os.path.join(self.checkpoint_path, "optimizer.pt")):
                self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "optimizer.pt")))
            if os.path.isfile(os.path.join(self.checkpoint_path, "scheduler.pt")):
                self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "scheduler.pt")))

        logger.info("Start training")
        logger.info("Epoch size = %d", self.n_epochs)
        logger.info("Batch size = %d", self.batch_size)
        logger.info("Global step = %d", global_step)

        total_loss = 0.0

        if self.rank == -1:
            self.model.zero_grad()
        else:
            self.model.module.zero_grad()

        for epoch in range(epochs_trained, self.n_epochs + 1):
            if self.early_stop:
                logger.info("Break at epoch {} and global step {}".format(epoch, global_step))
                break
            logger.info("Train epoch = %d", epoch)
            epoch_loss = 0.0
            self.model.train()
            if self.rank != -1:
                self.train_sampler.set_epoch(epoch)

            if self.use_tqdm and self.rank in [-1, 0]:
                progress = enumerate(tqdm(self.train_dataloader, desc="Iteration", leave=False), 1)
            else:
                progress = enumerate(self.train_dataloader, 1)

            for step, batch in progress:

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                move_dict_value_to_device(batch, device=self.device, rank=self.rank)
                if self.rank == -1:
                    loss = self.model.loss(batch, self.loss) / float(self.gradient_accumulation_steps)
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                else:
                    loss = self.model.module.loss(batch, self.loss) / float(self.gradient_accumulation_steps)
                    single_batch_loss = reduce_mean(loss, dist.get_world_size()).item()
                    epoch_loss += single_batch_loss
                    total_loss += single_batch_loss

                if self.rank in [-1, 0]:
                    self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=global_step)

                # 梯度反传
                if self.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # 参数更新
                if isinstance(self.gradient_accumulation_steps,
                              int) and global_step % self.gradient_accumulation_steps == 0:

                    # 梯度裁剪
                    if self.grad_norm:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_norm)
                        else:
                            utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                global_step += 1

                # 学习率更新
                if self.scheduler and isinstance(self.scheduler_steps, int) and global_step % self.scheduler_steps == 0:
                    self.scheduler.step()
                    # If there is one global learning rate (which is the common case).
                    lr = next(iter(self.optimizer.param_groups))['lr']
                    logger.info('Global step: {}, Learning rate: {}'.format(global_step, lr))
                    if self.rank in [-1, 0]:
                        self.writer.add_scalar(tag='Learning rate', scalar_value=lr, global_step=global_step)

                # 打印训练信息
                if isinstance(self.print_every, int) and global_step % self.print_every == 0:
                    logger.info('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'.
                                format(epoch, self.n_epochs, step, self.batch_count, loss.item()))

                # 保存模型
                if self.save_path and isinstance(self.save_steps,
                                                 int) and global_step % self.save_steps == 0 and self.rank in [-1, 0]:
                    logger.info("Saving models step = %d", global_step)
                    output_dir = os.path.join(self.save_path, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving models checkpoint to %s", output_dir)
                    model = self.model if self.rank == -1 else self.model.module
                    save_model(model=model, model_path=os.path.join(output_dir, "models.pt"))
                    # logger.info("Saving trainer arguments to %s", output_dir)
                    # save_json(vars(self), os.path.join(output_dir, "trainer.json"))
                    if self.optimizer:
                        logger.debug("Saving optimizer states to %s", output_dir)
                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    if self.scheduler:
                        logger.debug("Saving scheduler states to %s", output_dir)
                        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                # 验证模型
                if self.dev_data and isinstance(self.validate_steps,
                                                int) and global_step % self.validate_steps == 0 and self.rank in [-1,
                                                                                                                  0]:
                    logger.info("Evaluate step = %d", global_step)
                    model = self.model if self.rank == -1 else self.model.module
                    model.eval()
                    if self.use_tqdm:
                        progress = enumerate(tqdm(self.dev_dataloader, desc="Evaluating", leave=False, position=0), 1)
                    else:
                        progress = enumerate(self.dev_dataloader, 1)
                    with torch.no_grad():
                        for step, batch in progress:
                            move_dict_value_to_device(batch, device=self.device, rank=self.rank)
                            model.evaluate(batch, self.metrics)
                    self.model.train()
                    evaluate_result = self.metrics.get_metric()
                    logger.info("Evaluate result = %s", str(evaluate_result))
                    if self.rank in [-1, 0]:
                        for key, value in evaluate_result.items():
                            self.writer.add_scalar(tag=key, scalar_value=value, global_step=global_step)
                        # wandb.log({
                        #     "Global Step":global_step,
                        #     **evaluate_result,
                        # })
                    if self.best_metric["evaluate_result"] is None:
                        self.best_metric = {
                            "evaluate_result": evaluate_result,
                            "global_step": global_step,
                        }
                        if self.save_by_metric:
                            output_dir = os.path.join(self.save_best_model_path, "checkpoint-{}".format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model = self.model if self.rank == -1 else self.model.module
                            save_model(model=model, model_path=os.path.join(output_dir, "models.pt"))
                            if self.optimizer:
                                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            if self.scheduler:
                                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    else:
                        if self.metric_mode == 'max':
                            is_better = evaluate_result[self.metric_name] > self.best_metric["evaluate_result"][
                                self.metric_name]
                        else:
                            is_better = evaluate_result[self.metric_name] < self.best_metric["evaluate_result"][
                                self.metric_name]
                        if is_better:
                            curr_metric = {"evaluate_result": evaluate_result, "global_step": global_step}
                            prev_step = self.best_metric["global_step"]
                            curr_step = curr_metric["global_step"]
                            if self.save_by_metric is not None:
                                prev_output_dir = os.path.join(self.save_best_model_path,
                                                               "checkpoint-{}".format(prev_step))
                                shutil.rmtree(prev_output_dir)
                                output_dir = os.path.join(self.save_best_model_path, "checkpoint-{}".format(curr_step))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model = self.model if self.rank == -1 else self.model.module
                                save_model(model=model, model_path=os.path.join(output_dir, "models.pt"))
                                if self.optimizer:
                                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                if self.scheduler:
                                    torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                verb = "increases" if self.metric_mode == 'max' else 'decreases'
                                logger.info("Metric {} {} from {:.3f} to {:.3f},save models checkpoint to {}".format(
                                    self.metric_name,
                                    verb,
                                    self.best_metric["evaluate_result"][self.metric_name],
                                    curr_metric["evaluate_result"][self.save_by_metric],
                                    output_dir)
                                )
                            else:
                                verb = "increases" if self.metric_mode == 'max' else 'decreases'
                                logger.info("Metric {} {} from {:.3f} to {:.3f}.".format(
                                    self.metric_name,
                                    verb,
                                    self.best_metric["evaluate_result"][self.metric_name],
                                    curr_metric["evaluate_result"][self.metric_name],
                                ))
                            self.best_metric = curr_metric

                    if self.early_stopping:
                        if not self.early_stopping.metric_name:
                            self.early_stopping.metric_name = evaluate_result.keys()[0]
                        self.early_stopping(evaluate_result[self.early_stopping.metric_name])
                        if self.early_stopping.early_stop:
                            self.early_stop = True
                            logger.info("Early Stop with patience={},threshold={} on metric {}.".format(
                                self.early_stopping.patience, self.early_stopping.threshold,
                                self.early_stopping.metric_name,
                            ))
                            break

            logger.info("Epoch loss = %f", epoch_loss)

        logger.info("At global step {},the best evaluation results are:{}".format(
            self.best_metric["global_step"], self.best_metric["evaluate_result"]
        ))
        logger.info("End training")