#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Log metrics to tensorboard.

This file provides interface to log any metrics in tensorboard, could be
extended to any other tool like visdom.

.. code-block: none

   tensorboard --logdir <PARLAI_DATA/tensorboard> --port 8888.
"""

import datetime
import json
import numbers
import os
import pathlib
import re
from typing import Optional

import utils.logging as logging
from core.metrics import Metric, dict_report, get_metric_display_data
from core.opt import Opt
from core.params import ParlaiParser
from utils.io import PathManager

_TB_SUMMARY_INVALID_TAG_CHARACTERS = re.compile(r'[^-/\w\.]')


class TensorboardLogger(object):
    """
    Log objects to tensorboard.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add tensorboard CLI args.
        """
        logger = parser.add_argument_group('Tensorboard Arguments')
        logger.add_argument(
            '-tblog',
            '--tensorboard-log',
            type='bool',
            default=False,
            help="Tensorboard logging of metrics",
            hidden=False,
        )
        logger.add_argument(
            '-tblogdir',
            '--tensorboard-logdir',
            type=str,
            default=None,
            help="Tensorboard logging directory, defaults to model_file.tensorboard",
            hidden=False,
        )
        return parser

    def __init__(self, opt: Opt):
        try:
            # tensorboard is a very expensive thing to import. Wait until the
            # last second to import it.
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please run `pip install tensorboard tensorboardX`.')

        if opt['tensorboard_logdir'] is not None:
            tbpath = opt['tensorboard_logdir']
        else:
            tbpath = opt['model_file'] + '.tensorboard'

        logging.debug(f'Saving tensorboard logs to: {tbpath}')
        if not PathManager.exists(tbpath):
            PathManager.mkdirs(tbpath)
        self.writer = SummaryWriter(tbpath, comment=json.dumps(opt))

    def log_metrics(self, setting, step, report):
        """
        Log all metrics to tensorboard.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        for k, v in report.items():
            v = v.value() if isinstance(v, Metric) else v
            if not isinstance(v, numbers.Number):
                logging.error(f'k {k} v {v} is not a number')
                continue
            display = get_metric_display_data(metric=k)
            # Remove invalid characters for TensborboardX Summary beforehand
            # so that the logs aren't cluttered with warnings.
            tag = _TB_SUMMARY_INVALID_TAG_CHARACTERS.sub('_', f'{k}/{setting}')
            try:
                self.writer.add_scalar(
                    tag,
                    v,
                    global_step=step,
                    display_name=f"{display.title}",
                    summary_description=display.description,
                )
            except TypeError:
                # internal tensorboard doesn't support custom display titles etc
                self.writer.add_scalar(tag, v, global_step=step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    """
    Log objects to Weights and Biases.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add WandB CLI args.
        """
        logger = parser.add_argument_group('WandB Arguments')
        logger.add_argument(
            '-wblog',
            '--wandb-log',
            type='bool',
            default=False,
            help="Enable W&B logging of metrics",
        )
        logger.add_argument(
            '--wandb-name',
            type=str,
            default=None,
            help='W&B run name. If not set, WandB will randomly generate a name.',
            hidden=True,
        )

        logger.add_argument(
            '--wandb-project',
            type=str,
            default=None,
            help='W&B project name. Defaults to timestamp. Usually the name of the sweep.',
            hidden=False,
        )
        logger.add_argument(
            '--wandb-entity',
            type=str,
            default=None,
            help='W&B entity name.',
            hidden=False,
        )

        logger.add_argument(
            '--wandb-log-model',
            type=bool,
            default=False,
            help='Enable logging of model artifacts to weight and biases',
            hidden=False,
        )
        return logger

    def __init__(self, opt: Opt, model=None):
        try:
            # wand is a very expensive thing to import. Wait until the
            # last second to import it.
            import wandb

        except ImportError:
            raise ImportError('Please run `pip install wandb`.')

        name = opt.get('wandb_name')
        project = opt.get('wandb_project') or datetime.datetime.now().strftime(
            '%Y-%m-%d-%H-%M'
        )

        self.run = wandb.init(
            name=name,
            project=project,
            dir=os.path.dirname(opt['model_file']),
            notes=f"{opt['model_file']}",
            entity=opt.get('wandb_entity'),
            reinit=True,  # in case of preemption
            resume=True,  # requeued runs should be treated as single run
        )
        # suppress wandb's output
        logging.getLogger("wandb").setLevel(logging.ERROR)

        if not self.run.resumed:
            task_arg = opt.get("task", None)
            if task_arg:
                if (
                    len(task_arg.split(",")) == 1
                ):  # It gets confusing to parse these args for multitask teachers, so don't.
                    maybe_task_opts = task_arg.split(":")
                    for task_opt in maybe_task_opts:
                        if len(task_opt.split("=")) == 2:
                            k, v = task_opt.split("=")
                            setattr(self.run.config, k, v)

            for key, value in opt.items():
                if key not in self.run.config:  # set by task logic
                    if value is None or isinstance(value, (str, numbers.Number, tuple)):
                        setattr(self.run.config, key, value)

        self.model_file = opt.get('model_file', None)
        if model is not None:
            self.run.watch(model)

    def log_metrics(self, setting, step, report):
        """
        Log all metrics to W&B.

        :param setting:
            One of train/valid/test. Will be used as the title for the graph.
        :param step:
            Number of parleys
        :param report:
            The report to log
        """
        report = dict_report(report)
        report = {
            f'{k}/{setting}': v
            for k, v in report.items()
            if isinstance(v, numbers.Number)
        }
        report['custom_step'] = step
        self.run.log(report)

    def log_final(self, setting, report):
        report = dict_report(report)
        report = {
            f'{k}/{setting}': v
            for k, v in report.items()
            if isinstance(v, numbers.Number)
        }
        for key, value in report.items():
            self.run.summary[key] = value

    def log_model(self, model_file=None):
        if self.model_file is not None:
            if not model_file:
                model_file = self.model_file
            model_file = pathlib.Path(model_file)
            if model_file.exists():
                self.run.log_artifact(
                    str(model_file),
                    name=f"{self.run.name}-{model_file.name}",
                    type="model",
                )
            vocab_file = model_file.with_suffix('.dict')
            if vocab_file.exists():
                self.run.log_artifact(
                    str(vocab_file),
                    name=f"{self.run.name}-{vocab_file.name}",
                    type="vocab",
                )
            stats_file = model_file.with_suffix('.trainstats')
            if stats_file.exists():
                self.run.log_artifact(
                    str(stats_file),
                    name=f"{self.run.name}-{stats_file.name}",
                    type="stats",
                )

    def finish(self):
        self.run.finish()

    def flush(self):
        pass
