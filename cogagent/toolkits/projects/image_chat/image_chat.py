from __future__ import annotations
import argparse
import importlib
import os
import sys as _sys
import datetime

import copy
import json
import pickle
import traceback
import pkg_resources


from collections import defaultdict
from typing import List, Optional
from typing import Dict, Any
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


KEEP_ALL = 'all'
SHARED: Dict[str, Any] = {}
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    # silence the error
    GIT_AVAILABLE = False
# import parlai.utils.logging as logging
import os
import sys
import logging
from logging import getLogger  # noqa: F401

try:
    import coloredlogs

    COLORED_LOGS = True
except ImportError:
    COLORED_LOGS = False

SPAM = 5
DEBUG = logging.DEBUG
VERBOSE = DEBUG + 5
INFO = logging.INFO
REPORT = INFO + 5
SUCCESS = REPORT + 1
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(SPAM, "SPAM")
logging.addLevelName(REPORT, "REPORT")
logging.addLevelName(SUCCESS, "SUCCESS")

COLORED_FORMAT = '%(asctime)s | %(message)s'
CONSOLE_FORMAT = '%(asctime)s %(levelname).4s | %(message)s'
CONSOLE_DATE_FORMAT = '%H:%M:%S'
LOGFILE_FORMAT = '%(asctime)s %(levelname)-8s | %(message)s'
LOGFILE_DATE_FORMAT = None

COLORED_LEVEL_STYLES = {
    'spam': {'color': 'white', 'faint': True},
    'debug': {'color': 'green', 'faint': True},
    'verbose': {'color': 'blue'},
    'error': {'color': 'red'},
    'info': {},
    'report': {'bold': True},
    'success': {'bold': True, 'color': 'green'},
    'warning': {'color': 'yellow'},
    'critical': {'bold': True, 'color': 'red'},
}


def _is_interactive():
    if os.environ.get('PARLAI_FORCE_COLOR'):
        return True
    try:
        __IPYTHON__
        return True
    except NameError:
        return sys.stdout.isatty()


# Some functions in this class assume that ':' will be the separator used in
# the logging formats setup for this class
class ParlaiLogger(logging.Logger):
    def __init__(self, name, console_level=INFO):
        """
        Initialize the logger object.

        :param name:
            Name of the logger
        :param console_level:
            minimum level of messages logged to console
        """
        super().__init__(name, console_level)  # can be initialized with any level
        # Logging to stdout
        self.streamHandler = logging.StreamHandler(sys.stdout)
        # Log to stdout levels: console_level and above
        self.prefix = None
        self.interactive = _is_interactive()
        self.streamHandler.setFormatter(self._build_formatter())
        super().addHandler(self.streamHandler)

    def _build_formatter(self):
        prefix_format = f'{self.prefix} ' if self.prefix else ''
        if COLORED_LOGS and self.interactive:
            return coloredlogs.ColoredFormatter(
                prefix_format + COLORED_FORMAT,
                datefmt=CONSOLE_DATE_FORMAT,
                level_styles=COLORED_LEVEL_STYLES,
                field_styles={},
            )
        elif self.interactive:
            return logging.Formatter(
                prefix_format + CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT
            )
        else:
            return logging.Formatter(
                prefix_format + LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT
            )

    def force_interactive(self):
        self.interactive = True
        self.streamHandler.setFormatter(self._build_formatter())

    def log(self, msg, level=INFO):
        """
        Default Logging function.
        """
        super().log(level, msg)

    def add_format_prefix(self, prefix):
        """
        Include `prefix` in all future logging statements.
        """
        # change both handler formatters to add a prefix
        self.prefix = prefix
        self.streamHandler.setFormatter(self._build_formatter())

    def mute(self):
        """
        Stop logging to stdout.
        """
        self.prev_level = self.streamHandler.level
        self.streamHandler.level = ERROR
        return self.prev_level

    def unmute(self):
        """
        Resume logging to stdout.
        """
        self.streamHandler.level = self.prev_level


# -----------------------------------
# Forming the logger                #
# -----------------------------------
logger = ParlaiLogger(name="parlai")


def set_log_level(level):
    logger.setLevel(level)


def disable():
    logger.mute()


def enable():
    logger.unmute()


def info(msg):
    return logger.info(msg)


def critical(msg):
    return logger.critical(msg)


def report(msg):
    return logger.log(msg, level=REPORT)


def success(msg):
    return logger.log(msg, level=SUCCESS)


def log(*args, **kwargs):
    return logger.log(*args, **kwargs)


def verbose(msg):
    return logger.log(msg, level=VERBOSE)


def debug(*args, **kwargs):
    return logger.debug(*args, **kwargs)


def error(*args, **kwargs):
    return logger.error(*args, **kwargs)


def warn(*args, **kwargs):
    return logger.warning(*args, **kwargs)


def warning(*args, **kwargs):
    return logger.warning(*args, **kwargs)


def get_all_levels():
    levels = set(logging._nameToLevel.keys())
    levels.remove('WARNING')
    return [l.lower() for l in levels]

# from parlai.core.build_data import modelzoo_path
def modelzoo_path(datapath, path):
    """
    Map pretrain models filenames to their path on disk.

    If path starts with 'models:', then we remap it to the model zoo path within the
    data directory (default is ParlAI/data/models). We download models from the model
    zoo if they are not here yet.
    """
    if path is None:
        return None
    if (
        not path.startswith('models:')
        and not path.startswith('zoo:')
        and not path.startswith('izoo:')
    ):
        return path
    elif path.startswith('models:') or path.startswith('zoo:'):
        zoo = path.split(':')[0]
        zoo_len = len(zoo) + 1
        model_path = path[zoo_len:]
        # Check if we need to download the model
        if "/" in path:
            animal = path[zoo_len : path.rfind('/')].replace('/', '.')
        else:
            animal = path[zoo_len:]
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            my_module.download(datapath)
        except (ImportError, AttributeError):
            try:
                # maybe we didn't find a specific model, let's try generic .build
                animal_ = '.'.join(animal.split(".")[:-1]) + '.build'
                module_name_ = 'parlai.zoo.{}'.format(animal_)
                my_module = importlib.import_module(module_name_)
                my_module.download(datapath)
            except (ImportError, AttributeError) as exc:
                # truly give up
                raise ImportError(
                    f'Could not find pretrained model in {module_name} or {module_name_}.'
                    ' Please check your spelling and make sure you\'ve pulled from master.'
                ) from exc

        return os.path.join(datapath, 'models', model_path)
    else:
        # Internal path (starts with "izoo:") -- useful for non-public
        # projects.  Save the path to your internal model zoo in
        # parlai_internal/.internal_zoo_path
        # TODO: test the internal zoo.
        zoo_path = 'parlai_internal/zoo/.internal_zoo_path'
        if not PathManager.exists('parlai_internal/zoo/.internal_zoo_path'):
            raise RuntimeError(
                'Please specify the path to your internal zoo in the '
                'file parlai_internal/zoo/.internal_zoo_path in your '
                'internal repository.'
            )
        else:
            with PathManager.open(zoo_path, 'r') as f:
                zoo = f.read().split('\n')[0]
            return os.path.join(zoo, path[5:])



from typing import Callable, Dict, Type
import importlib

from collections import namedtuple

script_registration = namedtuple('script_registration', ('klass', 'hidden', 'aliases'))


AGENT_REGISTRY: Dict[str, Type] = {}
TEACHER_REGISTRY: Dict[str, Type] = {}
SCRIPT_REGISTRY: Dict[str, script_registration] = {}


def register_agent(name: str) -> Callable[[Type], Type]:
    """
    Register an agent to be available in command line calls.

    >>> @register_agent("my_agent")
    ... class MyAgent:
    ...     pass
    """

    def _inner(cls_):
        global AGENT_REGISTRY
        AGENT_REGISTRY[name] = cls_
        return cls_

    return _inner


def register_script(name: str, aliases=None, hidden=False):
    """
    Register an agent to be available in command line calls.

    >>> @register_script("my_script")
    ... class MyScript:
    ...     pass
    """
    if aliases is None:
        aliases = []

    def _inner(cls_):
        global SCRIPT_REGISTRY
        SCRIPT_REGISTRY[name] = script_registration(cls_, hidden, aliases)
        return cls_

    return _inner


def register_teacher(name: str) -> Callable[[Type], Type]:
    """
    Register a teacher to be available as a command line.

    >>> @register_teacher("my_teacher")
    ... class MyTeacher:
    ...    pass
    """

    def _inner(cls_):
        global TEACHER_REGISTRY
        TEACHER_REGISTRY[name] = cls_
        return cls_

    return _inner


##############################################################
### AGENT LOADER
##############################################################
def _name_to_agent_class(name: str):
    """
    Convert agent name to class.

    This adds "Agent" to the end of the name and uppercases the first letter
    and the first letter appearing after each underscore (underscores are
    removed).

    :param name:
        name of agent, e.g. local_human

    :return:
        class of agent, e.g. LocalHumanAgent.
    """
    words = name.split('_')
    class_name = ''
    for w in words:
        # capitalize the first letter
        class_name += w[0].upper() + w[1:]
    # add Agent to the end of the name
    class_name += 'Agent'
    return class_name


def load_agent_module(agent_path: str):
    """
    Return the module for an agent specified by ``--model``.

    Can be formatted in several different ways:

    * full: `-m parlai.agents.seq2seq.seq2seq:Seq2seqAgent`
    * shorthand: -m seq2seq, which will check both paths
      ``parlai.agents.seq2seq.seq2seq:Seq2seqAgent`` and
      ``parlai.agents.seq2seq.agents:Seq2seqAgent``
    * half-shorthand: ``-m seq2seq/variant``, which will check the path
      `parlai.agents.seq2seq.variant:VariantAgent`

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:seq2seq".

    To use agents in projects, you can prepend "projects:" and the name of the
    project folder to model arguments, e.g. "projects:personachat:kvmemnn"
    will translate to ``projects/personachat/kvmemnn``.

    :param agent_path:
        path to model class in one of the above formats.

    :return:
        module of agent
    """
    global AGENT_REGISTRY
    if agent_path in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_path]

    repo = 'parlai'
    if agent_path.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        # this will follow the same paths but look in parlai_internal instead
        repo = 'parlai_internal'
        agent_path = agent_path[9:]
    elif agent_path.startswith('fb:'):
        repo = 'parlai_fb'
        agent_path = agent_path[3:]

    if agent_path.startswith('projects:'):
        # e.g. -m projects:personachat:kvmemnn
        path_list = agent_path.split(':')
        if len(path_list) != 3:
            raise RuntimeError(
                'projects paths should follow pattern '
                'projects:folder:model; you used {}'
                ''.format(agent_path)
            )
        folder_name = path_list[1]
        model_name = path_list[2]
        module_name = 'projects.{p}.{m}.{m}'.format(m=model_name, p=folder_name)
        class_name = _name_to_agent_class(model_name)
    elif ':' in agent_path:
        # e.g. -m "parlai.agents.seq2seq.seq2seq:Seq2seqAgent"
        path_list = agent_path.split(':')
        module_name = path_list[0]
        class_name = path_list[1]
    elif '/' in agent_path:
        # e.g. -m my_agent/special_variant
        # will check parlai.agents.my_agent.special_variant:SpecialVariantAgent
        path_list = agent_path.split('/')
        module_name = "%s.agents.%s.%s" % (repo, path_list[0], path_list[1])
        class_name = _name_to_agent_class(path_list[1])
    else:
        # e.g. -m seq2seq
        # will check parlai.agents.seq2seq.agents for Seq2seqAgent first
        # then check parlai.agents.seq2seq.seq2seq for Seq2seqAgent second
        class_name = _name_to_agent_class(agent_path)
        try:
            module_name = "%s.agents.%s.agents" % (repo, agent_path)
            importlib.import_module(module_name)  # check if it's there
        except ImportError:
            module_name = "%s.agents.%s.%s" % (repo, agent_path, agent_path)

    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)

    return model_class


##############################################################
### TASK AND TEACHER LOADERS
##############################################################
def _get_task_path_and_repo(taskname: str):
    """
    Returns the task path list and repository containing the task as specified by
    `--task`.

    :param taskname: path to task class (specified in format detailed below)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        task = task[9:]
    elif task.startswith('fb:'):
        repo = 'parlai_fb'
        task = task[3:]

    task_path_list = task.split(':')

    return task_path_list, repo


def load_task_module(taskname: str):
    """
    Get the module containing all teacher agents for the task specified by `--task`.

    :param taskname:
        path to task class in one of the formats

    """
    task_path_list, repo = _get_task_path_and_repo(taskname)
    task_path = task_path_list[0]

    if '.' in task_path:
        module_name = task_path
    else:
        task = task_path.lower()
        module_name = "%s.tasks.%s.agents" % (repo, task)
        module_name = "%s.agents" %task

    task_module = importlib.import_module(module_name)

    return task_module


def load_teacher_module(taskname: str):
    """
    Get the module of the teacher agent specified by `--task`.

    Can be formatted in several different ways:

    * full: ``-t parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand: ``-t babi``, which will check
      ``parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand specific: ``-t babi:task10k``, which will check
      ``parlai.tasks.babi.agents:Task10kTeacher``

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:babi".

    Options can be sent to the teacher by adding an additional colon,
    for example ``-t babi:task10k:1`` directs the babi Task10kTeacher to use
    task number 1.

    :param taskname: path to task class in one of the above formats.

    :return:
        teacher module
    """
    global TEACHER_REGISTRY
    if taskname in TEACHER_REGISTRY:
        return TEACHER_REGISTRY[taskname]

    task_module = load_task_module(taskname)
    task_path_list, repo = _get_task_path_and_repo(taskname)

    if len(task_path_list) > 1 and '=' not in task_path_list[1]:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        teacher = task_path_list[1]
        if '.' not in task_path_list[0] and 'Teacher' not in teacher:
            # Reformat from underscore to CamelCase and append "Teacher" to
            # class name by default if a complete path is not given.
            words = teacher.split('_')
            teacher_name = ''
            for w in words:
                teacher_name += w[0].upper() + w[1:]
            teacher = teacher_name + "Teacher"
    else:
        teacher = "DefaultTeacher"

    teacher_class = getattr(task_module, teacher)
    return teacher_class





def load_world_module(
    taskname: str,
    interactive_task: bool = False,
    selfchat_task: bool = False,
    num_agents: int = None,  # a priori may not know the number of agents
    default_world=None,
):
    """
    Load the world module for the specific environment. If not enough information is to
    determine which world should be loaded, returns None.

    :param taskname:
        path to task class in one of the above formats
    :param interactive_task:
        whether or not the task is interactive
    :param num_agents:
        number of agents in the world; this may not be known a priori
    :param default_world:
        default world to return if specified

    :return:
        World module (or None, if not enough info to determine is present)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        task = task[9:]
    task_path_list = task.split(':')

    if len(task_path_list) > 1:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        world_name = task_path_list[1] + "World"
        if interactive_task:
            world_name = "Interactive" + world_name
        elif selfchat_task:
            world_name = "SelfChat" + world_name
    else:
        if interactive_task:
            world_name = "InteractiveWorld"
        elif selfchat_task:
            world_name = "SelfChatWorld"
        else:
            world_name = "DefaultWorld"

    if '.' in task_path_list[0]:
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly)
        module_name_parts = task_path_list[0].split('.')
        if module_name_parts[-1] == 'agents':
            # The world will be located in ".worlds", so correct the path
            module_name = '.'.join(module_name_parts[:-1]) + '.worlds'
        else:
            module_name = task_path_list[0]
    else:
        task = task_path_list[0].lower()
        module_name = f"{repo}.tasks.{task}.worlds"

    try:
        my_module = importlib.import_module(module_name)
        world_class = getattr(my_module, world_name)
    except (ModuleNotFoundError, AttributeError):
        # Defaults to this if you did not specify a world for your task.
        world_class = _get_default_world(default_world, num_agents)

    return world_class



# from parlai.tasks.tasks import ids_to_tasks
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Helper functions for defining the set of tasks in ParlAI.

The actual task list and definitions are in the file task_list.py
"""
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains a list of all the tasks, their id and task name, description and the
tags associated with them.
"""

task_list = [
    {
        "id": "AmazonQA",
        "display_name": "AmazonQA",
        "task": "amazon_qa",
        "tags": ["QA"],
        "links": {"website": "http://jmcauley.ucsd.edu/data/amazon/qa/"},
        "description": (
            "This dataset contains Question and Answer data from Amazon, "
            "totaling around 1.4 million answered questions."
        ),
    },
    {
        "id": "AQuA",
        "display_name": "AQuA",
        "task": "aqua",
        "tags": ["QA"],
        "links": {"arXiv": "https://arxiv.org/abs/1705.04146"},
        "description": (
            "Dataset containing algebraic word problems with rationales for "
            "their answers."
        ),
    },
    {
        "id": "bAbI-1k",
        "display_name": "bAbI 1k",
        "task": "babi:All1k",
        "tags": ["QA"],
        "description": (
            "20 synthetic tasks that each test a unique aspect of text and "
            "reasoning, and hence test different capabilities of learning "
            "models."
        ),
        "links": {"arXiv": "http://arxiv.org/abs/1502.05698"},
        "notes": (
            "You can access just one of the bAbI tasks with e.g. "
            "'babi:Task1k:3' for task 3."
        ),
    },
    {
        "id": "bAbI-10k",
        "display_name": "bAbI 10k",
        "task": "babi:All10k",
        "tags": ["QA"],
        "description": (
            "20 synthetic tasks that each test a unique aspect of text and "
            "reasoning, and hence test different capabilities of learning "
            "models."
        ),
        "links": {"arXiv": "http://arxiv.org/abs/1502.05698"},
        "notes": (
            "You can access just one of the bAbI tasks with e.g. 'babi:Task10k:3' "
            "for task 3."
        ),
    },
    {
        "id": "BlendedSkillTalk",
        "display_name": "Blended Skill Talk",
        "task": "blended_skill_talk",
        "tags": ["ChitChat"],
        "description": (
            "A dataset of 7k conversations explicitly designed to exhibit multiple "
            "conversation modes: displaying personality, having empathy, and "
            "demonstrating knowledge."
        ),
    },
    {
        "id": "BookTest",
        "display_name": "BookTest",
        "task": "booktest",
        "tags": ["Cloze"],
        "description": (
            "Sentence completion given a few sentences as context from a book. "
            "A larger version of CBT."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1610.00956"},
    },
    {
        "id": "BotAdversarialDialogue",
        "display_name": "Bot Adversarial Dialogue ",
        "task": "bot_adversarial_dialogue",
        "tags": [],
        "description": (
            "Datasets described in the paper Recipes for Safety in Open-domain Chatbots. "
            "Datasets consist of classification tasks in which the goal is to "
            "determine if the utterance is offensive or not given a dialogue context. "
        ),
        "links": {"arXiv": "https://arxiv.org/abs/2010.07079"},
    },
    {
        "id": "SafetyMix",
        "display_name": "Safety Mix",
        "task": "safety_mix",
        "tags": [],
        "description": (
            "Datasets described in the paper: Learning from data in the mixed adversarial non-adversarial case:"
            "Finding the helpers and ignoring the trolls. "
            "Datasets based on Bot Adversarial Dialogue and consist of a mixture of different troll users."
            "Artificial noise is introduced to the dataset given the troll user type."
        ),
    },
    {
        "id": "CBT",
        "display_name": "Children's Book Test (CBT)",
        "task": "cbt",
        "tags": ["Cloze"],
        "description": (
            "Sentence completion given a few sentences as context from a "
            "children's book."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1511.02301"},
    },
    {
        "id": "CCPE",
        "display_name": "Coached Conversational Preference Elicitation",
        "task": "ccpe",
        "tags": ["Goal"],
        "description": (
            "A dataset consisting of 502 dialogs with 12,000 annotated "
            "utterances between a user and an assistant discussing movie "
            "preferences in natural language. It was collected using a "
            "Wizard-of-Oz methodology between two paid crowd-workers, "
            "where one worker plays the role of an 'assistant', while "
            "the other plays the role of a 'user'."
        ),
        "links": {
            "website": "https://ai.google/tools/datasets/coached-conversational-preference-elicitation"
        },
    },
    {
        "id": "CMU_DoG",
        "display_name": "CMU Document Grounded Conversations",
        "task": "cmu_dog",
        "tags": ["ChitChat", "Grounded"],
        "description": (
            "A document grounded dataset for text conversations, where the "
            "documents are Wikipedia articles about popular movies. Consists "
            "of 4112 conversations with an average of 21.43 turns per conversation."
        ),
        "links": {
            "arXiv": "https://arxiv.org/abs/1809.07358",
            "github": "https://github.com/festvox/datasets-CMU_DoG",
        },
    },
    {
        "id": "COPA",
        "display_name": "Choice of Plausible Alternatives",
        "task": "copa",
        "tags": ["Reasoning"],
        "description": (
            "The Choice Of Plausible Alternatives (COPA) evaluation provides "
            "researchers with a tool for assessing progress in open-domain "
            "commonsense causal reasoning. COPA consists of 1000 questions, "
            "split equally into development and test sets of 500 questions each."
        ),
        "links": {"website": "http://people.ict.usc.edu/~gordon/copa.html"},
    },
    {
        "id": "COQA",
        "display_name": "Conversational Question Answering Challenge",
        "task": "coqa",
        "tags": ["QA"],
        "description": (
            "CoQA is a large-scale dataset for building Conversational "
            "Question Answering systems. The goal of the CoQA challenge "
            "is to measure the ability of machines to understand a text "
            "passage and answer a series of interconnected questions that "
            "appear in a conversation. CoQA is pronounced as coca."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1808.07042"},
    },
    {
        "id": "CornellMovie",
        "display_name": "Cornell Movie",
        "task": "cornell_movie",
        "tags": ["ChitChat", "Dodeca"],
        "description": ("Fictional conversations extracted from raw movie scripts."),
        "links": {"arXiv": "https://arxiv.org/abs/1106.3077"},
    },
    {
        "id": "DBLL-bAbI",
        "display_name": "Dialog Based Language Learning: bAbI Task",
        "task": "dbll_babi",
        "tags": ["Goal"],
        "description": (
            "Short dialogs based on the bAbI tasks, but in the form of a "
            "question from a teacher, the answer from the student, and finally a "
            "comment on the answer from the teacher. The aim is to find learning "
            "models that use the comments to improve."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1604.06045"},
        "notes": (
            "Tasks can be accessed with a "
            "format like: 'parlai display_data -t "
            "dbll_babi:task:2_p0.5' which specifies task 2, and policy with 0.5 "
            "answers correct, see the paper for more details of the tasks."
        ),
    },
    {
        "id": "DBLL-Movie",
        "display_name": "Dialog Based Language Learning: WikiMovies Task",
        "task": "dbll_movie",
        "tags": ["Goal"],
        "description": (
            "Short dialogs based on WikiMovies, but in the form of a question "
            "from a teacher, the answer from the student, and finally a comment "
            "on the answer from the teacher. The aim is to find learning models "
            "that use the comments to improve."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1604.06045"},
    },
    {
        "id": "dialog-bAbI",
        "display_name": "Dialog bAbI",
        "task": "dialog_babi",
        "tags": ["Goal"],
        "description": "Simulated dialogs of restaurant booking",
        "links": {"arXiv": "https://arxiv.org/abs/1605.07683"},
    },
    {
        "id": "dialog-bAbI-plus",
        "display_name": "Dialog bAbI+",
        "task": "dialog_babi_plus",
        "tags": ["Goal"],
        "description": (
            "bAbI+ is an extension of the bAbI Task 1 dialogues with everyday "
            "incremental dialogue phenomena (hesitations, restarts, and "
            "corrections) which model the disfluencies and communication "
            "problems in everyday spoken interaction in real-world environments. "
        ),
        "links": {
            "website": (
                "https://www.researchgate.net/publication/"
                "319128941_Challenging_Neural_Dialogue_Models_with_Natural_"
                "Data_Memory_Networks_Fail_on_Incremental_Phenomena"
            ),
            "paper": "http://aclweb.org/anthology/D17-1235",
        },
    },
    {
        "id": "dialogue-nli",
        "display_name": "Dialogue NLI",
        "task": "dialogue_nli",
        "tags": ["ChitChat", "NLI"],
        "description": (
            "Dialogue NLI is a dataset that addresses the issue of consistency in "
            "dialogue models."
        ),
        "links": {
            "website": "https://wellecks.github.io/dialogue_nli/",
            "arXiv": "https://arxiv.org/abs/1811.00671",
        },
    },
    {
        "id": "dstc7",
        "display_name": "DSTC7 subtrack 1 - ubuntu",
        "task": "dstc7",
        "tags": ["ChitChat"],
        "description": (
            "DSTC7 is a competition which provided a dataset of dialogs very "
            "similar to the ubuntu dataset. In particular, the subtrack 1 "
            "consists in predicting the next utterance."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1901.03461"},
    },
    {
        "id": "FVQA",
        "display_name": "FVQA",
        "task": "fvqa",
        "tags": ["Visual"],
        "description": (
            "The FVQA, a VQA dataset which requires, and supports, much deeper "
            "reasoning. We extend a conventional visual question answering "
            "dataset, which contains image-question-answer triplets, through "
            "additional image-question-answer-supporting fact tuples. The "
            "supporting fact is represented as a structural triplet, such as "
            "<Cat,CapableOf,ClimbingTrees>."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1606.05433"},
    },
    {
        "id": "DealNoDeal",
        "display_name": "Deal or No Deal",
        "task": "dealnodeal",
        "tags": ["Negotiation"],
        "description": (
            "End-to-end negotiation task which requires two agents to agree on "
            "how to divide a set of items, with each agent assigning different "
            "values to each item."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1706.05125"},
    },
    {
        "id": "Friends",
        "display_name": "Friends",
        "task": "friends",
        "tags": ["MultiPartyConvo"],
        "description": (
            "Multi-party conversation dataset modified from the 10 seasons "
            "of the popular American sitcom that ran in the 90s, Friends."
        ),
        "links": {"website": "https://convokit.cornell.edu/documentation/friends.html"},
    },
    {
        "id": "Glue",
        "display_name": "Glue",
        "task": "glue",
        "tags": [],
        "description": (
            "GLUE, the General Language Understanding Evaluation benchmark is "
            "a collection of resources for training, evaluating, and analyzing "
            "natural language understanding systems."
        ),
        "links": {
            "website": "https://gluebenchmark.com/",
            "website2": "https://huggingface.co/datasets/glue",
        },
    },
    {
        "id": "HotpotQA",
        "display_name": "HotpotQA",
        "task": "hotpotqa",
        "tags": ["QA"],
        "description": (
            "HotpotQA is a dataset for multi-hop question answering. "
            "The overall setting is that given some context paragraphs"
            "(e.g., a few paragraphs, or the entire Web) and a question,"
            "a QA system answers the question by extracting a span of text"
            "from the context. It is necessary to perform multi-hop reasoning"
            "to correctly answer the question."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1809.09600"},
    },
    {
        "id": "HuggingFace",
        "display_name": "HuggingFace",
        "task": "huggingface",
        "tags": [],
        "description": ("HuggingFace datasets"),
        "links": {"website": "https://huggingface.co/"},
    },
    {
        "id": "LIGHT-Dialogue",
        "display_name": "LIGHT-Dialogue",
        "task": "light_dialog",
        "tags": ["Grounded", "Dodeca"],
        "description": (
            "LIGHT is a text adventure game with actions and dialogue collected. "
            "The source data is collected between crowdworkers playing the game."
        ),
        "links": {
            "website": "http://parl.ai/projects/light",
            "arXiv": "https://arxiv.org/abs/1903.03094",
        },
    },
    {
        "id": "LIGHT-Dialogue-Wild",
        "display_name": "LIGHT-Dialogue-Wild",
        "task": "light_dialog_wild",
        "tags": ["Grounded", "LIGHT"],
        "description": (
            " LIGHT is a text adventure game with actions and dialogue. "
            "The WILD dataset here features 41,131+ training episodes of dialogue "
            "collected from deploying a game as described in "
        ),
        "links": {
            "arXiv": "https://arxiv.org/abs/2008.08076",
            "website": "http://parl.ai/projects/light",
        },
    },
    {
        "id": "MutualFriends",
        "display_name": "MutualFriends",
        "task": "mutualfriends",
        "tags": ["Goal"],
        "description": (
            "Task where two agents must discover which friend of theirs is "
            "mutual based on the friends's attributes."
        ),
        "links": {"website": "https://stanfordnlp.github.io/cocoa/"},
    },
    {
        "id": "MCTest",
        "display_name": "MCTest",
        "task": "mctest",
        "tags": ["QA"],
        "description": ("Questions about short children's stories."),
        "links": {
            "website": (
                "https://www.microsoft.com/en-us/research/publication/"
                "mctest-challenge-dataset-open-domain-machine-comprehension-text/"
            )
        },
    },
    {
        "id": "MovieDD-QA",
        "display_name": "Movie Dialog QA",
        "task": "moviedialog:Task:1",
        "tags": ["QA", "MovieDD"],
        "description": (
            "Closed-domain QA dataset asking templated questions about movies, "
            "answerable from Wikipedia, similar to WikiMovies."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1511.06931"},
    },
    {
        "id": "MovieDD-QARecs",
        "display_name": "Movie Dialog QA Recommendations",
        "task": "moviedialog:Task:3",
        "tags": ["Goal", "MovieDD"],
        "description": (
            "Dialogs discussing questions about movies as well as recommendations."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1511.06931"},
    },
    {
        "id": "MovieDD-Recs",
        "display_name": "Movie Dialog Recommendations",
        "task": "moviedialog:Task:2",
        "tags": ["QA", "MovieDD"],
        "description": ("Questions asking for movie recommendations."),
        "links": {"arXiv": "https://arxiv.org/abs/1511.06931"},
    },
    {
        "id": "MovieDD-Reddit",
        "display_name": "Movie Dialog Reddit",
        "task": "moviedialog:Task:4",
        "tags": ["ChitChat", "MovieDD"],
        "description": (
            "Dialogs discussing Movies from Reddit (the Movies SubReddit)."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1511.06931"},
    },
    {
        "id": "MTurkWikiMovies",
        "display_name": "MTurk WikiMovies",
        "task": "mturkwikimovies",
        "tags": ["QA"],
        "description": (
            "Closed-domain QA dataset asking MTurk-derived questions about "
            "movies, answerable from Wikipedia."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1611.09823"},
    },
    {
        "id": "MultiNLI",
        "display_name": "MultiNLI",
        "task": "multinli",
        "tags": ["Entailment", "decanlp"],
        "description": (
            "A dataset designed for use in the development and evaluation of "
            "machine learning models for sentence understanding. Each example "
            "contains a premise and hypothesis. Model has to predict whether "
            "premise and hypothesis entail, contradict or are neutral to each "
            "other."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1704.05426"},
    },
    {
        "id": "NarrativeQA",
        "display_name": "NarrativeQA",
        "task": "narrative_qa",
        "tags": ["QA"],
        "description": (
            "A dataset and set of tasks in which the reader must answer "
            "questions about stories by reading entire books or movie scripts. "
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1712.07040"},
        "notes": (
            "You can access summaries only task for NarrativeQA by using task "
            "'narrative_qa:summaries'. By default, only stories are provided."
        ),
    },
    {
        "id": "NaturalQuestions",
        "display_name": "Natural Questions",
        "task": "natural_questions",
        "tags": ["QA"],
        "description": (
            "An open domain question answering dataset. "
            "Each example contains real questions that people searched "
            "for in Google and the content of the a Wikipedia article that "
            "was amongst the top 5 search resutls for that query, "
            "and its annotations. The annotations have the options of a long "
            "answer that is seleced from span of major content entities in "
            "the Wikipedia article (e.g., paragraphs, tables), a short answer"
            "that is selected from one or more short span of words in the "
            "article, or 'yes/no'. The existence of any of these answer "
            "formats depends on whether the main question can be answered, "
            "given the article; if not they are left empty."
        ),
        "links": {
            "paper": "https://research.google/pubs/pub47761/",
            "website": "https://ai.google.com/research/NaturalQuestions",
        },
        "notes": (
            "Since this task uses ChunkTeacher, it should be used with streaming."
        ),
    },
    {
        "id": "OpenSubtitles",
        "display_name": "Open Subtitles",
        "task": "opensubtitles",
        "tags": ["ChitChat"],
        "description": "Dataset of dialogs from movie scripts.",
        "links": {
            "version 2018 website": "http://opus.lingfil.uu.se/OpenSubtitles2018.php",
            "version 2009 website": "http://opus.lingfil.uu.se/OpenSubtitles.php",
            "related work (arXiv)": "https://arxiv.org/abs/1506.05869",
        },
    },
    {
        "id": "personalized-dialog-full",
        "display_name": "Personalized Dialog Full Set",
        "task": "personalized_dialog:AllFull",
        "tags": ["Goal", "Personalization"],
        "description": (
            "Simulated dataset of restaurant booking focused on personalization "
            "based on user profiles."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1706.07503"},
    },
    {
        "id": "personalized-dialog-small",
        "display_name": "Personalized Dialog Small Set",
        "task": "personalized_dialog:AllSmall",
        "tags": ["Goal", "Personalization"],
        "description": (
            "Simulated dataset of restaurant booking focused on personalization "
            "based on user profiles."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1706.07503"},
    },
    {
        "id": "prosocial_dialog",
        "display_name": "Prosocial Dialog",
        "task": "prosocial_dialog",
        "tags": [],
        "description": (
            "Prosocial Dialog dataset of 58K dialogues between a speaker showing "
            "potentially unsafe behavior and a speaker giving constructive feedback "
            "for more socially acceptable behavior."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/2205.12688"},
    },
    {
        "id": "QACNN",
        "display_name": "QA CNN",
        "task": "qacnn",
        "tags": ["Cloze"],
        "description": (
            "Cloze dataset based on a missing (anonymized) entity phrase from a "
            "CNN article"
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1506.03340"},
    },
    {
        "id": "QADailyMail",
        "display_name": "QA Daily Mail",
        "task": "qadailymail",
        "tags": ["Cloze"],
        "description": (
            "Cloze dataset based on a missing (anonymized) entity phrase from a "
            "Daily Mail article."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1506.03340"},
    },
    {
        "id": "QuAC",
        "display_name": "Question Answering in Context",
        "task": "quac",
        "tags": ["QA"],
        "description": (
            "Question Answering in Context is a dataset for modeling, "
            "understanding, and participating in information seeking dialog. Data "
            "instances consist of an interactive dialog between two crowd workers: "
            "(1) a student who poses a sequence of freeform questions to learn as "
            "much as possible about a hidden Wikipedia text, and (2) a teacher who "
            "answers the questions by providing short excerpts (spans) from the text. "
            "QuAC introduces challenges not found in existing machine comprehension "
            "datasets: its questions are often more open-ended, unanswerable, "
            "or only meaningful within the dialog context."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1808.07036"},
    },
    {
        "id": "SimpleQuestions",
        "display_name": "Simple Questions",
        "task": "simplequestions",
        "tags": ["QA"],
        "description": ("Open-domain QA dataset based on Freebase triples."),
        "links": {"arXiv": "https://arxiv.org/abs/1506.02075"},
    },
    {
        "id": "SNLI",
        "display_name": "The Stanford Natural Language Inference (SNLI) Corpus",
        "task": "snli",
        "tags": ["Entailment"],
        "description": (
            "The SNLI corpus (version 1.0) is a collection of 570k "
            "human-written English sentence pairs manually labeled for balanced "
            "classification with the labels entailment, contradiction, and "
            "neutral, supporting the task of natural language inference (NLI), "
            "also known as recognizing textual entailment (RTE)"
        ),
        "links": {"website": "https://nlp.stanford.edu/projects/snli/"},
    },
    {
        "id": "SQuAD2",
        "display_name": "SQuAD2",
        "task": "squad2",
        "tags": ["QA"],
        "description": (
            "Open-domain QA dataset answerable from a given paragraph from "
            "Wikipedia."
        ),
        "links": {"arXiv": "http://arxiv.org/abs/1806.03822"},
    },
    {
        "id": "SQuAD",
        "display_name": "SQuAD",
        "task": "squad",
        "tags": ["QA"],
        "description": (
            "Open-domain QA dataset answerable from a given paragraph from "
            "Wikipedia."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1606.05250"},
    },
    {
        "id": "SuperGLUE",
        "display_name": "SuperGLUE",
        "task": "superglue",
        "tags": [],
        "description": (
            "SuperGLUE (https://super.gluebenchmark.com/) is a new benchmark "
            "styled after GLUE with a new set of more difficult language "
            "understanding tasks, improved resources, and a new public "
            "leaderboard."
        ),
        "links": {
            "website": "https://super.gluebenchmark.com/",
            "website2": "https://huggingface.co/datasets/super_glue",
        },
    },
    {
        "id": "TriviaQA",
        "display_name": "TriviaQA",
        "task": "triviaqa",
        "tags": ["QA"],
        "description": (
            "Open-domain QA dataset with question-answer-evidence triples."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1705.03551"},
    },
    {
        "id": "TaskNTalk",
        "display_name": "Task N' Talk",
        "task": "taskntalk",
        "tags": ["Goal"],
        "description": (
            "Dataset of synthetic shapes described by attributes, for agents to "
            "play a cooperative QA game."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1706.08502"},
    },
    {
        "id": "Ubuntu",
        "display_name": "Ubuntu",
        "task": "ubuntu",
        "tags": ["ChitChat", "Dodeca"],
        "description": (
            "Dialogs between an Ubuntu user and an expert trying to fix issue, "
            "we use the V2 version, which cleaned the data to some extent. "
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1506.08909"},
    },
    {
        "id": "WebQuestions",
        "display_name": "Web Questions",
        "task": "webquestions",
        "tags": ["QA"],
        "description": ("Open-domain QA dataset from Web queries."),
        "links": {"paper": "http://www.aclweb.org/anthology/D13-1160"},
    },
    {
        "id": "WikiMovies",
        "display_name": "WikiMovies",
        "task": "wikimovies",
        "tags": ["QA"],
        "description": (
            "Closed-domain QA dataset asking templated questions about movies, "
            "answerable from Wikipedia."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1606.03126"},
    },
    {
        "id": "WikiQA",
        "display_name": "WikiQA",
        "task": "wikiqa",
        "tags": ["QA"],
        "description": ("Open domain QA from Wikipedia dataset"),
        "links": {
            "website": (
                "https://www.microsoft.com/en-us/research/publication/wikiqa-a-"
                "challenge-dataset-for-open-domain-question-answering/"
            )
        },
    },
    {
        "id": "VQAv1",
        "display_name": "VQAv1",
        "task": "vqa_v1",
        "tags": ["Visual"],
        "description": ("Open-ended question answering about visual content."),
        "links": {"arXiv": "https://arxiv.org/abs/1505.00468"},
    },
    {
        "id": "VQAv2",
        "display_name": "VQAv2",
        "task": "vqa_v2",
        "tags": ["Visual"],
        "description": ("Bigger, more balanced version of the original VQA dataset."),
        "links": {"arXiv": "https://arxiv.org/abs/1612.00837"},
    },
    {
        "id": "VisDial",
        "display_name": "VisDial",
        "task": "visdial",
        "tags": ["Visual"],
        "description": (
            "Task which requires agents to hold a meaningful dialog about "
            "visual content."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1611.08669"},
    },
    {
        "id": "MNIST_QA",
        "display_name": "MNIST_QA",
        "task": "mnist_qa",
        "tags": ["Visual"],
        "description": (
            "Task which requires agents to identify which number they are "
            "seeing. From the MNIST dataset."
        ),
    },
    {
        "id": "InsuranceQA",
        "display_name": "InsuranceQA",
        "task": "insuranceqa",
        "tags": ["QA"],
        "description": (
            "Task which requires agents to identify high quality answers "
            "composed by professionals with deep domain knowledge."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1508.01585"},
    },
    {
        "id": "MS_MARCO",
        "display_name": "MS_MARCO",
        "task": "ms_marco",
        "tags": ["QA"],
        "description": (
            "A large scale Machine Reading Comprehension Dataset with questions "
            "sampled from real anonymized user queries and contexts from web "
            "documents."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1611.09268"},
    },
    {
        "id": "CLEVR",
        "display_name": "CLEVR",
        "task": "clevr",
        "tags": ["Visual"],
        "description": (
            "A visual reasoning dataset that tests abilities such as attribute "
            "identification, counting, comparison, spatial relationships, and "
            "logical operations."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1612.06890"},
    },
    {
        "id": "nlvr",
        "display_name": "nlvr",
        "task": "nlvr",
        "tags": ["Visual"],
        "description": (
            "Cornell Natural Language Visual Reasoning (NLVR) is a language "
            "grounding dataset based on  pairs of natural language statements "
            "grounded in synthetic images."
        ),
        "links": {"website": "http://lic.nlp.cornell.edu/nlvr/"},
    },
    {
        "id": "WMT",
        "display_name": "WMT",
        "task": "wmt",
        "tags": ["MT"],
        "description": (
            "Workshop on Machine Translation task, currently only includes en_de."
        ),
    },
    {
        "id": "IWSLT14",
        "display_name": "IWSLT14",
        "task": "iwslt14",
        "tags": ["MT", "decanlp"],
        "description": (
            "2014 International Workshop on Spoken Language task, currently "
            "only includes en_de and de_en."
        ),
        "links": {"website": "https://wit3.fbk.eu"},
    },
    {
        "id": "ConvAI2",
        "display_name": "ConvAI2",
        "task": "convai2",
        "tags": ["ChitChat", "Dodeca"],
        "description": (
            "A chit-chat dataset based on PersonaChat for a NIPS 2018 competition. "
        ),
        "links": {
            "arXiv": "https://arxiv.org/abs/1801.07243",
            "website": "http://convai.io/",
        },
    },
    {
        "id": "ConvAI_ChitChat",
        "display_name": "ConvAI_ChitChat",
        "task": "convai_chitchat",
        "tags": ["ChitChat", "decanlp"],
        "description": (
            "Human-bot dialogues containing free discussions of randomly chosen "
            "paragraphs from SQuAD."
        ),
        "links": {"website": "http://convai.io/data/"},
    },
    {
        "id": "Dialogue_QE",
        "display_name": "Dialogue_QE",
        "task": "dialogue_qe",
        "tags": [],
        "description": (
            "Human-bot dialogues labelled for quality at the level of "
            "dialogues. Can be used to train dialogue-level metric for dialogue "
            "systems."
        ),
    },
    {
        "id": "QAngaroo",
        "display_name": "QAngaroo",
        "task": "qangaroo",
        "tags": ["QA"],
        "description": (
            "Reading Comprehension with Multiple Hop. Including two datasets: "
            "WIKIHOP built on on wikipedia, MEDHOP built on paper abstracts from "
            "PubMed."
        ),
        "links": {"website": "http://qangaroo.cs.ucl.ac.uk/"},
    },
    {
        "id": "SCAN",
        "display_name": "SCAN",
        "task": "scan",
        "tags": ["Goal"],
        "description": (
            "SCAN is a set of simple language-driven navigation tasks for "
            "studying compositional learning and zero-shot generalization. The "
            "SCAN tasks were inspired by the CommAI environment, which is the "
            "origin of the acronym (Simplified versions of the CommAI Navigation "
            "tasks)."
        ),
        "links": {
            "arXiv": "https://arxiv.org/abs/1711.00350",
            "website": "https://github.com/brendenlake/SCAN",
        },
    },
    {
        "id": "Persona-Chat",
        "display_name": "Persona-Chat",
        "task": "personachat",
        "tags": ["ChitChat"],
        "description": (
            "A chit-chat dataset where paired Turkers are given assigned "
            "personas and chat to try to get to know each other."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1801.07243"},
    },
    {
        "id": "TaskMaster",
        "display_name": "TaskMaster-1-2019",
        "task": "taskmaster",
        "tags": ["ChitChat"],
        "description": (
            "A chit-chat dataset by GoogleAI providing high quality goal-oriented conversations"
            "The dataset hopes to provoke interest in written vs spoken language"
            "Both the datasets consists of two-person dialogs:"
            "Spoken: Created using Wizard of Oz methodology. "
            "Written: Created by crowdsourced workers who were asked to write the "
            "full conversation themselves playing roles of both the user and assistant."
        ),
        "links": {"website": "https://ai.google/tools/datasets/taskmaster-1"},
    },
    {
        "id": "MSR-E2E",
        "display_name": "MSR End-to-End",
        "task": "msr_e2e",
        "tags": ["ChitChat"],
        "description": (
            "MSR-E2E is a dataset of human-human conversations in which one "
            "human plays the role of an Agent and the other one plays the role"
            "of a User. Data is collected from Amazon Mechanical Turk. "
        ),
        "links": {"website": "https://github.com/xiul-msr/e2e_dialog_challenge"},
    },
    {
        "id": "Twitter",
        "display_name": "Twitter",
        "task": "twitter",
        "tags": ["ChitChat", "Dodeca"],
        "description": (
            "Twitter data found on GitHub. No "
            "train/valid/test split was provided so 10k for valid and 10k for "
            "test was chosen at random."
        ),
        "links": {"website": "https://github.com/Marsan-Ma/chat_corpus/"},
    },
    {
        "id": "Wikipedia",
        "display_name": "Wikipedia",
        "task": 'wikipedia',
        "tags": [],
        "description": ("Dump of Wikipedia articles from 2/3/18"),
        "notes": (
            "Specify ':full' for the full articles to be returned, otherwise "
            "defaults to ':summary', which provides the first paragraphs. To put "
            "the article in the labels and the title in the text, specify "
            "':key-value' at the end (for a title/content key-value "
            "association)"
        ),
    },
    {
        "id": "Flickr30k",
        "display_name": "Flickr30k",
        "task": "flickr30k",
        "tags": ["Visual"],
        "description": ("30k captioned images pulled from Flickr compiled by UIUC. "),
        "links": {
            "website": "http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/",
            "paper1": "https://arxiv.org/abs/1505.04870v2",
            "paper2": "http://aclweb.org/anthology/Q14-1006",
        },
    },
    {
        "id": "COCO_Captions",
        "display_name": "COCO_Captions",
        "task": "coco_caption",
        "tags": ["Visual"],
        "description": (
            "COCO annotations derived from the 2015 COCO Caption Competition. "
        ),
        "links": {"website": "http://cocodataset.org/"},
    },
    {
        "id": "integration_tests",
        "display_name": "Integration Tests",
        "task": "integration_tests",
        "tags": ["Debug"],
        "description": ("Artificial tasks for ensuring models perform as expected"),
    },
    {
        "id": "ConvAI2_wild_evaluation",
        "display_name": "ConvAI2_wild_evaluation",
        "task": "convai2_wild_evaluation",
        "tags": ["ChitChat"],
        "description": (
            "Dataset collected during the wild evaluation of ConvaAI2 participants "
            "bots. 60% train, 20% valid and 20% test is chosen at "
            "random from the whole dataset."
        ),
        "links": {"website": "http://convai.io"},
    },
    {
        "id": "sst",
        "display_name": "SST Sentiment Analysis",
        "task": "sst",
        "tags": ["decanlp"],
        "description": (
            "Dataset containing sentiment trees of movie reviews. We use the modified "
            "binary sentence analysis subtask given by the DecaNLP paper here."
        ),
        "links": {
            "website": "https://nlp.stanford.edu/sentiment/index.html",
            "website2": "https://github.com/openai/generating-reviews-discovering-sentiment/",
        },
    },
    {
        "id": "cnn_dm",
        "display_name": "CNN/DM Summarisation",
        "task": "cnn_dm",
        "tags": ["decanlp"],
        "description": (
            "Dataset collected from CNN and the Daily Mail with summaries as labels, "
            "Implemented as part of the DecaNLP task."
        ),
        "links": {"website": "https://cs.nyu.edu/~kcho/DMQA/"},
    },
    {
        "id": "qasrl",
        "display_name": "QA-SRL Semantic Role Labeling",
        "task": "qasrl",
        "tags": ["decanlp"],
        "description": ("QA dataset implemented as part of the DecaNLP task."),
        "links": {"website": "https://dada.cs.washington.edu/qasrl/"},
    },
    {
        "id": "qazre",
        "display_name": "QA-ZRE Relation Extraction",
        "task": "qazre",
        "tags": ["decanlp"],
        "description": (
            "Zero Shot relation extraction task implemented as part of the DecaNLP "
            "task."
        ),
        "links": {"website": "http://nlp.cs.washington.edu/zeroshot/"},
    },
    {
        "id": "woz",
        "display_name": "WOZ restuarant reservation (Goal-Oriented Dialogue)",
        "task": "woz",
        "tags": ["decanlp"],
        "description": (
            "Dataset containing dialogues dengotiating a resturant reservation. "
            "Implemented as part of the DecaNLP task, focused on the change "
            "in the dialogue state."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1604.04562"},
    },
    {
        "id": "wikisql",
        "display_name": "WikiSQL semantic parsing task",
        "task": "wikisql",
        "tags": ["decanlp"],
        "description": (
            "Dataset for parsing sentences to SQL code, given a table. "
            "Implemented as part of the DecaNLP task."
        ),
        "links": {"website": "https://github.com/salesforce/WikiSQL"},
    },
    {
        "id": "mwsc",
        "display_name": "MWSC pronoun resolution",
        "task": "mwsc",
        "tags": ["decanlp"],
        "description": (
            "Resolving possible ambiguous pronouns. "
            "Implemented as part of the DecaNLP "
            "task, and can be found on the decaNLP github."
        ),
        "links": {"website": "https://github.com/salesforce/decaNLP"},
    },
    {
        "id": "decanlp",
        "display_name": "DecaNLP: The Natural Language Decathlon",
        "task": "decanlp",
        "tags": [],
        "description": (
            "A collection of 10 tasks (SQuAD, IWSLT, CNN/DM, MNLI, SST, QA‑SRL,"
            "QA‑ZRE, WOZ, WikiSQL and MWSC) designed to challenge a model with a range "
            "of different tasks. Note that we use IWSLT 2014 instead of "
            "2016/2013test/2014test for train/dev/test as given in the DecaNLP paper. "
        ),
        "links": {
            "arXiv": "https://arxiv.org/abs/1806.08730",
            "github": "https://github.com/salesforce/decaNLP",
        },
    },
    {
        "id": "Personality_Captions",
        "display_name": "Personality_Captions",
        "task": "personality_captions",
        "tags": ["Visual"],
        "description": (
            "200k images from the YFCC100m dataset "
            "with captions conditioned on one of 215 personalities."
        ),
        "links": {
            "website": "https://multimediacommons.wordpress.com/yfcc100m-core-dataset/",
            "arXiv": "https://arxiv.org/abs/1810.10665",
        },
        "notes": (
            "If you have already downloaded the images, please specify with "
            "the `--yfcc-path` flag, as the image download script takes a "
            "very long time to run"
        ),
    },
    {
        "id": "Image_Chat",
        "display_name": "Image_Chat",
        "task": "image_chat",
        "tags": ["Visual", "ChitChat"],
        "description": (
            "202k dialogues and 401k utterances over 202k images from "
            "the YFCC100m dataset "
            "using 215 possible personality traits"
        ),
        "links": {
            "website": "https://klshuster.github.io/image_chat/",
            "website2": "https://multimediacommons.wordpress.com/yfcc100m-core-dataset/",
        },
        "notes": (
            "If you have already downloaded the images, please specify with "
            "the `--yfcc-path` flag, as the image download script takes a "
            "very long time to run"
        ),
    },
    {
        "id": "Image_Chat_Generation",
        "display_name": "Image_Chat_Generation",
        "task": "image_chat:Generation",
        "tags": ["Visual", "ChitChat", "Dodeca"],
        "description": ("Image Chat task to train generative model"),
    },
    {
        "id": "Wizard_of_Wikipedia",
        "display_name": "Wizard_of_Wikipedia",
        "task": "wizard_of_wikipedia",
        "tags": ["ChitChat"],
        "description": (
            "A dataset with conversations directly grounded with knowledge "
            "retrieved from Wikipedia. Contains 201k utterances from 22k "
            "dialogues spanning over 1300 diverse topics, split into train, "
            "test, and valid sets. The test and valid sets are split "
            "into two sets each: one with overlapping topics with the train "
            "set, and one with unseen topics."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1811.01241"},
        "notes": (
            "To access the different valid/test splits (unseen/seen), specify "
            "the corresponding split (`random_split` for seen, `topic_split` "
            "for unseen) after the last colon in the task. "
            "E.g. `wizard_of_wikipedia:WizardDialogKnowledgeTeacher:random_split`"
        ),
    },
    {
        "id": "Wizard_of_Wikipedia_Generator",
        "display_name": "Wizard_of_Wikipedia_Generator",
        "task": "wizard_of_wikipedia:Generator",
        "tags": ["ChitChat", "Dodeca"],
        "description": ("Wizard of Wikipedia task to train generative models"),
    },
    {
        "id": "DailyDialog",
        "display_name": "Daily Dialog",
        "task": "dailydialog",
        "tags": ["ChitChat", "Dodeca"],
        "description": (
            "A dataset of chitchat dialogues with strong annotations for "
            "topic, emotion and utterance act. This version contains both sides "
            "of every conversation, and uses the official train/valid/test splits "
            "from the original authors."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1710.03957"},
    },
    {
        "id": "EmpatheticDialogues",
        "display_name": "Empathetic Dialogues",
        "task": "empathetic_dialogues",
        "tags": ["ChitChat", "Dodeca"],
        "description": (
            "A dataset of 25k conversations grounded in emotional situations "
            "to facilitate training and evaluating dialogue systems. "
            "Dataset has been released under the CC BY-NC license."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1811.00207"},
        "notes": (
            "EmpatheticDialoguesTeacher returns examples like so: \n\n"
            "  - [text]:  context line (previous utterance by 'speaker') \n"
            "  - [labels]: label line  (current utterance by 'listener') \n\n"
            "with additional task specific fields: \n\n"
            "  - [situation]: a 1-3 sentence description of the situation that the conversation is \n"
            "  - [emotion]: one of 32 emotion words \n\n"
            "Other optional fields: \n\n"
            "  - [prepend_ctx]: fasttext prediction on context line - or None \n"
            "  - [prepend_cand]: fasttext prediction on label line (candidate) - or None \n"
            "  - [deepmoji_ctx]: vector encoding from deepmoji penultimate layer - or None \n"
            "  - [deepmoji_cand]: vector encoding from deepmoji penultimate layer for label line (candidate) - or None "
        ),
    },
    {
        "id": "DialogueSafety",
        "display_name": "Dialogue Safety",
        "task": "dialogue_safety",
        "tags": [],
        "description": (
            "Several datasets described in the paper Built it Break it Fix it "
            "for Dialogue Safety: Robustness from Adversarial Human Attack. "
            "All datasets are classification tasks in which the goal is to "
            "determine if the text is offensive or 'safe'."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1908.06083"},
    },
    {
        "id": "MultiDoGo",
        "display_name": "MultiDoGo",
        "task": "multidogo",
        "tags": ["TOD"],
        "description": (
            "MultiDoGo is a large task-oriented dataset from Amazon collected "
            "in a Wizard of Oz fashion, using both crowd and expert annotators "
            "with annotations at varying levels of granularity."
        ),
        "links": {
            "website": "https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset"
        },
    },
    {
        "id": "MultiWOZv2.0",
        "display_name": "MultiWOZ 2.0",
        "task": "multiwoz_v20",
        "tags": ["Goal"],
        "description": (
            "A fully labeled collection of human-written conversations spanning"
            "over multiple domains and topics."
        ),
        "links": {"website": "http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/"},
    },
    {
        "id": "MultiWOZv2.1",
        "display_name": "MultiWOZ 2.1",
        "task": "multiwoz_v21",
        "tags": ["Goal"],
        "description": (
            "A fully labeled collection of human-written conversations spanning"
            "over multiple domains and topics."
        ),
        "links": {"website": "http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/"},
    },
    {
        "id": "MultiWOZv2.2",
        "display_name": "MultiWOZ 2.2",
        "task": "multiwoz_v22",
        "tags": ["Goal"],
        "description": (
            "A fully labeled collection of human-written conversations spanning"
            "over multiple domains and topics. Schemas are included."
        ),
        "links": {
            "website": "https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2"
        },
    },
    {
        "id": "SelfChat",
        "display_name": "SelfChat",
        "task": "self_chat",
        "tags": [],
        "description": "Not a dataset, but a generic world for model self-chats.",
    },
    {
        "id": "OneCommon",
        "display_name": "OneCommon",
        "task": "onecommon",
        "tags": ["Goal"],
        "description": (
            "A collaborative referring task which requires advanced skills "
            "of common grounding under continuous and partially-observable context. "
            "This code also includes reference-resolution annotation."
        ),
        "links": {"website": "https://github.com/Alab-NII/onecommon"},
    },
    {
        "id": "IGC",
        "display_name": "Image Grounded Conversations",
        "task": "igc",
        "tags": ["Visual", "ChitChat", "Dodeca"],
        "description": (
            "A dataset of (image, context, question, answer) tuples, comprised "
            "of eventful images taken from Bing, Flickr, and COCO."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/1701.08251"},
    },
    {
        "id": "ANLI",
        "display_name": "Adversarial Natural Language Inference (ANLI) Corpus",
        "task": "anli",
        "tags": ["Entailment", "NLI"],
        "description": (
            "The ANLI corpus (version 1.0) is a new large-scale NLI benchmark dataset,"
            "collected via an iterative, adversarial human-and-model-in-the-loop procedure"
            "with the labels entailment, contradiction, and neutral. A total of three rounds "
            "of data are collected that progressively increase in difficulty and complexity."
        ),
        "links": {
            "github": "https://github.com/facebookresearch/anli",
            "arXiv": "https://arxiv.org/abs/1910.14599",
        },
    },
    {
        "id": "NLI",
        "display_name": "Natural Language Inference (NLI) Corpus",
        "task": "nli",
        "tags": ["Entailment"],
        "description": (
            "A collection of 3 popular Natural Language Inference(NLI) benchmark tasks: "
            "ANLI v0.1, MultiNLI 1.0, SNLI 1.0."
        ),
    },
    {
        "id": "Funpedia",
        "display_name": "Funpedia",
        "task": "funpedia",
        "tags": [],
        "description": (
            "Task for rephrasing sentences from Wikipedia conditioned on a persona."
        ),
    },
    {
        "id": "LIGHTGenderBias",
        "display_name": "LIGHT Gender Bias",
        "task": "light_genderation_bias",
        "tags": [],
        "description": ("Task for debiasing the LIGHT dataset."),
        "links": {"arXiv": "https://arxiv.org/abs/1911.03842"},
    },
    {
        "id": "AirDialogue",
        "display_name": "AirDialogue",
        "task": "airdialogue",
        "tags": ["Goal"],
        "description": (
            "Task for goal-oriented dialogue using airplane booking conversations "
            "between agents and customers."
        ),
        "links": {"website": "https://github.com/google/airdialogue"},
    },
    {
        "id": "HollE",
        "display_name": "Holl-E",
        "task": "holl_e",
        "tags": ["ChitChat"],
        "description": (
            "Sequence of utterances and responses with background knowledge about"
            "movies. From the Holl-E dataset."
        ),
        "links": {"website": "https://github.com/nikitacs16/Holl-E"},
    },
    {
        "id": "ELI5",
        "display_name": "ELI5",
        "task": "eli5",
        "tags": ["QA"],
        "description": (
            "This dataset contains Question and Answer data from Reddit "
            "explainlikeimfive posts and comments."
        ),
        "links": {"website": "https://github.com/facebookresearch/ELI5/"},
    },
    {
        "id": "ReDial",
        "display_name": "ReDial",
        "task": "redial",
        "tags": ["ChitChat", "Goal"],
        "description": (
            "Annotated dataset of dialogues where users recommend movies to each other."
        ),
        "links": {"website": "https://redialdata.github.io/website/"},
    },
    {
        "id": "DREAM",
        "display_name": "DREAM",
        "task": "dream",
        "tags": ["QA"],
        "description": (
            "A multiple-choice answering dataset based on multi-turn, multi-party dialogue."
        ),
        "links": {"website": "https://dataset.org/dream/"},
    },
    {
        "id": "C3",
        "display_name": "C3",
        "task": "c3",
        "tags": ["QA"],
        "description": (
            "A multiple-choice answering dataset in Chinese based on a prior passage."
        ),
        "links": {"website": "https://dataset.org/c3/"},
    },
    {
        "id": "CommonSenseQA",
        "display_name": "CommonSenseQA",
        "task": "commonsenseqa",
        "tags": ["QA"],
        "description": (
            "CommonSenseQA is a multiple-choice Q-A dataset that relies on commonsense "
            "knowlegde to predict correct answers."
        ),
        "links": {"wesite": "https://www.tau-nlp.org/commonsenseqa"},
    },
    {
        "id": "StyleGen",
        "display_name": "Style-Controlled Generation",
        "task": "style_gen",
        "tags": ["ChitChat"],
        "description": (
            "Dialogue datasets (BlendedSkillTalk, ConvAI2, EmpatheticDialogues, and "
            "Wizard of Wikipedia) labeled with personalities taken from the Image-Chat "
            "dataset. Used for the style-controlled generation project"
        ),
    },
    {
        "id": "GoogleSGD",
        "display_name": "GoogleSGD",
        "task": "google_sgd",
        "tags": ["Goal"],
        "description": (
            "The Schema-Guided Dialogue (SGD) dataset consists of over 20k "
            "annotated multi-domain, task-oriented conversations between a "
            "human and a virtual assistant."
        ),
    },
    {
        "id": "GoogleSGDSimulationSplits",
        "display_name": "GoogleSGD Simulation Splits",
        "task": "google_sgd_simulation_splits",
        "tags": ["Goal"],
        "description": (
            "Custom processing of the Google SGD dataset into In-Domain and "
            "Out-of-Domain splits for the use of zero and few-shotting with "
            "other task-oriented data."
        ),
    },
    {
        "id": "TaskMaster2",
        "display_name": "TaskMaster2",
        "task": "taskmaster2",
        "tags": ["Goal"],
        "description": (
            "The second version of TaskMaster, containing Wizard-of-Oz dialogues "
            "for task oriented dialogue in 7 domains."
        ),
    },
    {
        "id": "TaskMaster3",
        "display_name": "TicketTalk (Taskmaster3)",
        "task": "taskmaster3",
        "tags": ["Goal"],
        "description": (
            "Taskmaster3 is a dataset of movie ticket dialogues collected in a self-chat manner. To induce conversational"
            + "variety, crowd workers were asked to generate conversations given dozens of different instructions of"
            + "different level of specificity, some purposefully including conversational  errors."
        ),
    },
    {
        "id": "GenderationBiasControlTask",
        "display_name": "GenderationBiasControlTask",
        "task": "genderation_bias:controllable_task",
        "tags": [],
        "description": (
            "A teacher that wraps other ParlAI tasks and appends control tokens to the "
            "text field indicating the presence of gender words in the label(s)."
        ),
    },
    {
        "id": "MDGender",
        "display_name": "MD Gender",
        "task": "md_gender",
        "tags": [],
        "description": (
            "Tasks for the multi-dimensional gender bias classifier training."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/2005.00614"},
    },
    {
        "id": "Sensitive Topics Evaluation Topics Valid Teacher",
        "display_name": "Sensitive Topics Evaluation Topics Valid Teacher",
        "task": "sensitive_topics_evaluation",
        "tags": [],
        "description": (
            "Task for evaluating a classifier trained to identify conversational messages "
            "on the following sensitive topics: Politics, Drugs, Medical Advice, Religion, "
            "Relationships & Dating / NSFW."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/2010.07079"},
    },
    {
        "id": "decode",
        "display_name": "DialoguE COntradiction DEteCtion (DECODE)",
        "task": "decode",
        "tags": ["ChitChat", "Entailment"],
        "description": "Task for detect whether the last utterance contradicts previous dialogue history.",
        "links": {"arXiv": "https://arxiv.org/abs/2012.13391"},
    },
    {
        "id": "metalwoz",
        "display_name": "MetaLWOz",
        "task": "metalwoz",
        "tags": ["Goal"],
        "description": (
            "Meta-Learning Wizard-of-Oz (MetaLWOz) is a dataset designed to help "
            "develop models capable of predicting user responses in unseen domains."
        ),
        "links": {
            "paper": "http://workshop.colips.org/dstc7/dstc8/DTSC8_multidomain_task_proposal.pdf",
            "website": "https://www.microsoft.com/en-us/research/project/metalwoz/",
        },
    },
    {
        "id": "Wizard_of_Internet",
        "display_name": "Wizard_of_Internet",
        "task": "wizard_of_internet",
        "tags": ["ChitChat"],
        "description": (
            "A dataset with conversations directly grounded with knowledge "
            "retrieved from internet. One of the participants has access to internet search. "
            "The other side has an assigned persona that provides the topic of the conversation. "
            "Contains 93.7k utterances from 9.6k conversations, split into train, "
            "test, and valid sets."
        ),
    },
    {
        "id": "msc",
        "display_name": "MultiSessionChat",
        "task": "msc",
        "tags": ["ChitChat"],
        "description": (
            "A multi-session human-human chit-chat dataset consist of session 2-5 follow up from PersonaChat "
            "It contains 5k full converesations from session 2 to session 5 (session 1 being PersonaChat) "
        ),
    },
    {
        "id": "jericho_world",
        "display_name": "JerichoWorld",
        "task": "jericho_world",
        "tags": [],
        "description": (
            "Jericho World dataset: common sense in a text-based game. "
            "The goal is generating the knowledge graph of the game state "
            "or the set of valid actions from the text descriptions of the world."
        ),
    },
    {
        "id": "CaSiNo",
        "display_name": "CaSiNo (CampSite Negotiation Dialogues)",
        "task": "casino",
        "tags": ["Negotiation"],
        "description": (
            "A dataset of 1030 negotiation dialogues. Two participants take the role of campsite neighbors and negotiate for"
            + "Food, Water, and Firewood packages, based on their individual preferences and requirements."
        ),
        "links": {
            "paper": "https://aclanthology.org/2021.naacl-main.254.pdf",
            "website": "https://github.com/kushalchawla/CaSiNo",
        },
    },
    {
        "id": "SaFeRDialogues",
        "display_name": "SaFeRDialogues",
        "task": "saferdialogues",
        "tags": [],
        "description": (
            "A dataset of 8k dialogues demonstrating safety failures, feedback "
            "signaling them, and a response acknowledging the feedback. "
            "Dataset has been released under the CC BY-NC license."
        ),
        "links": {"arXiv": "https://arxiv.org/abs/2110.07518"},
    },
    {
        "id": "XPersona",
        "display_name": "XPersona",
        "task": "xpersona",
        "tags": ["ChitChat"],
        "description": (
            "XPersona is an extension of ConvAI2 with six more languages: Chinese, French, Indonesian, Italian, Korean, and Japanese."
        ),
        "links": {
            "arXiv": "https://arxiv.org/pdf/2003.07568.pdf",
            "website": "https://github.com/HLTCHKUST/Xpersona",
        },
    },
    {
        "id": "LCCC",
        "display_name": "LCCC",
        "task": "lccc",
        "tags": ["ChitChat"],
        "description": ("Large-scale cleaned Chinese conversation dataset."),
        "links": {
            "arXiv": "https://arxiv.org/pdf/2008.03946",
            "website": "https://github.com/thu-coai/CDial-GPT",
        },
    },
    {
        "id": "SPOLIN",
        "display_name": "SPOLIN",
        "task": "spolin",
        "tags": ["all", "engaging", "improv", "open-ended", "common ground"],
        "description": "Conversation pairs from the SPOLIN dataset. The pairs abide by the Yes-and principle of"
        + "improvisational theatre (improv).",
        "links": {
            "arXiv": "https://arxiv.org/abs/2004.09544",
            "website": "https://justin-cho.com/spolin",
        },
    },
    {
        "id": "Feedback for Interactive Talk & Search",
        "display_name": "FITS",
        "task": "fits",
        "tags": ["all", "engaging", "improve", "open-ended"],
        "description": "A human-model dialogue dataset consist of 14k dialogues where human speakers give feedbacks on bot responses.",
    },
    {
        "id": "EntailmentBank",
        "display_name": "EntailmentBank",
        "task": "entailment_bank",
        "tags": ["Entailment", "Reasoning"],
        "links": {
            "paper": "https://aclanthology.org/2021.emnlp-main.585.pdf",
            "website": "https://allenai.org/data/entailmentbank",
        },
        "description": (
            "2k multi-step entailment trees, explaining the answers to ARC science questions."
        ),
    },
    {
        "id": "ASDIV",
        "display_name": "ASDIV",
        "task": "asdiv",
        "tags": ["Math", "Reasoning"],
        "links": {
            "paper": "https://aclanthology.org/2020.acl-main.92.pdf",
            "website": "https://github.com/chaochun/nlu-asdiv-dataset",
        },
        "description": (
            "A diverse corpus for evaluating and developing English math Wword problem solvers."
        ),
    },
    {
        "id": "MathDataset",
        "display_name": "MathDataset",
        "task": "math_dataset",
        "tags": ["Math", "Reasoning"],
        "links": {
            "paper": "https://arxiv.org/pdf/2103.03874.pdf",
            "website": "https://github.com/hendrycks/math/",
        },
        "description": ("12,500 challenging competition mathematics problems."),
    },
    {
        "id": "EQASC",
        "display_name": "EQASC",
        "task": "eqasc",
        "tags": ["QA", "Reasoning"],
        "links": {
            "paper": "https://aclanthology.org/2020.emnlp-main.10.pdf",
            "website": "https://github.com/harsh19/Reasoning-Chains-MultihopQA",
        },
        "description": ("Reasoning chains for multihop question-answering set QASC."),
    },
    {
        "id": "ReasoningFramework",
        "display_name": "Reasoning Framework",
        "task": "reasoning",
        "tags": ["Reasoning"],
        "description": ("Reasoning teacher framework."),
    },
    {
        "id": "ProofWriter",
        "display_name": "ProofWriter",
        "task": "proof_writer",
        "tags": ["Reasoning"],
        "links": {
            "paper": "https://aclanthology.org/2021.findings-acl.317.pdf",
            "website": "https://allenai.org/data/proofwriter",
        },
        "description": (
            "A synthentically dataset of initial clauses and rules, with questions about statements that these initial"
            + "clauses and rules imply."
        ),
    },
]



def _preprocess(name):
    return name.lower().replace('-', '')


def _build(task_list):
    tasks = {}
    tags = defaultdict(list)

    for t in task_list:
        task = _preprocess(t['id'])
        tasks[task] = [t]
        for j in t['tags']:
            tag = _preprocess(j)
            if tag in tasks:
                raise RuntimeError('tag ' + tag + ' is the same as a task name')
            tags[tag].append(t)
    return tasks, tags


def _id_to_task_data(t_id):
    t_id = _preprocess(t_id)
    if t_id in tasks:
        # return the task assoicated with this task id
        return tasks[t_id]
    elif t_id in tags:
        # return the list of tasks for this tag
        return tags[t_id]
    else:
        # should already be in task form
        raise RuntimeError('could not find tag/task id')


def _id_to_task(t_id):
    if t_id[0] == '#':
        # this is a tag, so return all the tasks for this tag
        return ','.join((d['task'] for d in _id_to_task_data(t_id[1:])))
    else:
        # this should already be in task form
        return t_id


def ids_to_tasks(ids):
    if ids is None:
        raise RuntimeError(
            'No task specified. Please select a task with ' + '--task {task_name}.'
        )
    return ','.join((_id_to_task(i) for i in ids.split(',') if len(i) > 0))


# Build the task list from the json file.
tasks, tags = _build(task_list)

# from parlai.core.opt import Opt
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Opt is the system for passing around options throughout ParlAI.
"""

# these keys are automatically removed upon save. This is a rather blunt hammer.
# It's preferred you indicate this at option definition time.
__AUTOCLEAN_KEYS__: List[str] = [
    "override",
    "batchindex",
    "download_path",
    "datapath",
    "verbose",
    # we don't save interactive mode or load from checkpoint, it's only decided by scripts or CLI
    "interactive_mode",
    "load_from_checkpoint",
]


class Opt(dict):
    """
    Class for tracking options.

    Functions like a dict, but allows us to track the history of arguments as they are
    set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack(limit=2)[-2]
        self.history.append((key, val, loc))
        super().__setitem__(key, val)

    def __getstate__(self):
        return (self.history, self.deepcopies, dict(self))

    def __setstate__(self, state):
        self.history, self.deepcopies, data = state
        self.update(data)

    def __reduce__(self):
        return (Opt, (), self.__getstate__())

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that history is copied over to new object.
        """
        # track location of deepcopy
        loc = traceback.format_stack(limit=3)[-3]
        self.deepcopies.append(loc)
        # copy all our children
        memo = Opt({k: copy.deepcopy(v) for k, v in self.items()})
        # deepcopy the history. history is only tuples, so we can do it shallow
        memo.history = copy.copy(self.history)
        # deepcopy the list of deepcopies. also shallow bc only strings
        memo.deepcopies = copy.copy(self.deepcopies)
        return memo

    def display_deepcopies(self):
        """
        Display all deepcopies.
        """
        if len(self.deepcopies) == 0:
            return 'No deepcopies performed on this opt.'
        return '\n'.join(f'{i}. {loc}' for i, loc in enumerate(self.deepcopies, 1))

    def display_history(self, key):
        """
        Display the history for an item in the dict.
        """
        changes = []
        i = 0
        for key_, val, loc in self.history:
            if key != key_:
                continue
            i += 1
            changes.append(f'{i}. {key} was set to {val} at:\n{loc}')
        if changes:
            return '\n'.join(changes)
        else:
            return f'No history for {key}'

    def save(self, filename: str) -> None:
        """
        Save the opt to disk.

        Attempts to 'clean up' any residual values automatically.
        """
        # start with a shallow copy
        dct = dict(self)

        # clean up some things we probably don't want to save
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]

        with PathManager.open(filename, 'w', encoding='utf-8') as f:
            json.dump(dct, fp=f, indent=4)
            # extra newline for convenience of working with jq
            f.write('\n')

    @classmethod
    def load(cls, optfile: str) -> Opt:
        """
        Load an Opt from disk.
        """
        try:
            # try json first
            with PathManager.open(optfile, 'r', encoding='utf-8') as t_handle:
                dct = json.load(t_handle)
        except UnicodeDecodeError:
            # oops it's pickled
            with PathManager.open(optfile, 'rb') as b_handle:
                dct = pickle.load(b_handle)
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]
        return cls(dct)

    @classmethod
    def load_init(cls, optfile: str) -> Opt:
        """
        Like load, but also looks in opt_presets folders.

        optfile may also be a comma-separated list of multiple presets/files.
        """
        if "," in optfile:
            # load and combine each of the individual files
            new_opt = cls()
            for subopt in optfile.split(","):
                new_opt.update(cls.load_init(subopt))
            return new_opt

        oa_filename = os.path.join("opt_presets", optfile + ".opt")
        user_filename = os.path.join(os.path.expanduser(f"~/.parlai"), oa_filename)
        if PathManager.exists(optfile):
            return cls.load(optfile)
        elif PathManager.exists(user_filename):
            # use a user's custom opt preset
            return cls.load(user_filename)
        else:
            # Maybe a bundled opt preset
            for root in ['parlai', 'parlai_internal', 'parlai_fb']:
                try:
                    if pkg_resources.resource_exists(root, oa_filename):
                        return cls.load(
                            pkg_resources.resource_filename(root, oa_filename)
                        )
                except ModuleNotFoundError:
                    continue

        # made it through without a return path so raise the error
        raise FileNotFoundError(
            f"Could not find filename '{optfile} or opt preset '{optfile}.opt'. "
            "Please check https://parl.ai/docs/opt_presets.html for a list "
            "of available opt presets."
        )

    def log(self, header="Opt"):
        # from parlai.core.params import print_git_commit

        logging.info(header + ":")
        for key in sorted(self.keys()):
            valstr = str(self[key])
            if valstr.replace(" ", "").replace("\n", "") != valstr:
                # show newlines as escaped keys, whitespace with quotes, etc
                valstr = repr(valstr)
            logging.info(f"    {key}: {valstr}")
        print_git_commit()



# from parlai.utils.io import PathManager

try:
    from iopath.common.file_io import PathManager as _PathManager
except ImportError:
    try:
        from fvcore.common.file_io import PathManagerBase as _PathManager
    except ImportError:
        raise ImportError(
            "parlai now requires iopath for some I/O operations. Please run "
            "`pip install iopath`"
        )

USE_ATOMIC_TORCH_SAVE = True

PathManager = _PathManager()

try:
    # register any internal file handlers
    import parlai_fb  # noqa: F401

    # internal file handlers can't handle atomic saving. see T71772714
    USE_ATOMIC_TORCH_SAVE = not parlai_fb.finalize_registration(PathManager)
except ModuleNotFoundError:
    USE_ATOMIC_TORCH_SAVE = True


def print_git_commit():
    """
    Print the current git commit of ParlAI and parlai_internal.
    """
    if not GIT_AVAILABLE:
        return
    root = os.path.dirname(os.path.dirname(parlai.__file__))
    internal_root = os.path.join(root, 'parlai_internal')
    fb_root = os.path.join(root, 'parlai_fb')
    try:
        git_ = git.Git(root)
        current_commit = git_.rev_parse('HEAD')
        logging.info(f'Current ParlAI commit: {current_commit}')
    except git.GitCommandNotFound:
        pass
    except git.GitCommandError:
        pass

    try:
        git_ = git.Git(internal_root)
        internal_commit = git_.rev_parse('HEAD')
        logging.info(f'Current internal commit: {internal_commit}')
    except git.GitCommandNotFound:
        pass
    except git.GitCommandError:
        pass

    try:
        git_ = git.Git(fb_root)
        fb_commit = git_.rev_parse('HEAD')
        logging.info(f'Current fb commit: {fb_commit}')
    except git.GitCommandNotFound:
        pass
    except git.GitCommandError:
        pass


def print_announcements(opt):
    """
    Output any announcements the ParlAI team wishes to make to users.

    Also gives the user the option to suppress the output.
    """
    # no annoucements to make right now
    return

    noannounce_file = os.path.join(opt.get('datapath'), 'noannouncements')
    if PathManager.exists(noannounce_file):
        # user has suppressed announcements, don't do anything
        return

    # useful constants
    # all of these colors are bolded
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[1;91m'
    YELLOW = '\033[1;93m'
    GREEN = '\033[1;92m'
    BLUE = '\033[1;96m'
    CYAN = '\033[1;94m'
    MAGENTA = '\033[1;95m'

    # only use colors if we're outputting to a terminal
    USE_COLORS = _sys.stdout.isatty()
    if not USE_COLORS:
        RESET = BOLD = RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = ''

    # generate the rainbow stars
    rainbow = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]
    size = 78 // len(rainbow)
    stars = ''.join([color + '*' * size for color in rainbow])
    stars += RESET

    # do the actual output
    print(
        '\n'.join(
            [
                '',
                stars,
                BOLD,
                'Announcements go here.',
                RESET,
                # don't bold the suppression command
                'To suppress this message (and future announcements), run\n`touch {}`'.format(
                    noannounce_file
                ),
                stars,
            ]
        )
    )


def get_model_name(opt):
    """
    Get the model name from either `--model` or `--model-file`.
    """
    model = opt.get('model', None)
    if model is None:
        # try to get model name from model opt file
        model_file = opt.get('model_file', None)
        if model_file is not None:
            model_file = modelzoo_path(opt.get('datapath'), model_file)
            optfile = model_file + '.opt'
            if PathManager.exists(optfile):
                new_opt = Opt.load(optfile)
                model = new_opt.get('model', None)
    return model


def str2none(value: str):
    """
    If the value is a variant of `none`, return None.

    Otherwise, return the original value.
    """
    if value.lower() == 'none':
        return None
    else:
        return value


def str2bool(value):
    """
    Convert 'yes', 'false', '1', etc.

    into a boolean.
    """
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2floats(s):
    """
    Look for single float or comma-separated floats.
    """
    return tuple(float(f) for f in s.split(','))


def str2multitask_weights(s):
    if s == 'stochastic':
        return s
    else:
        return str2floats(s)


def str2class(value):
    """
    From import path string, returns the class specified.

    For example, the string
    'parlai.agents.hugging_face.dict:Gpt2DictionaryAgent' returns
    <class 'parlai.agents.hugging_face.dict.Gpt2DictionaryAgent'>.
    """
    if ':' not in value:
        raise RuntimeError('Use a colon before the name of the class.')
    name = value.split(':')
    module = importlib.import_module(name[0])
    return getattr(module, name[1])


def class2str(value):
    """
    Inverse of params.str2class().
    """
    s = str(value)
    s = s[s.find('\'') + 1 : s.rfind('\'')]  # pull out import path
    s = ':'.join(s.rsplit('.', 1))  # replace last period with ':'
    return s


def fix_underscores(args):
    """
    Convert underscores to hyphens in args.

    For example, converts '--gradient_clip' to '--gradient-clip'.

    :param args: iterable, possibly containing args strings with underscores.
    """
    if args:
        new_args = []
        for a in args:
            if type(a) is str and a.startswith('-'):
                a = a.replace('_', '-')
            new_args.append(a)
        args = new_args
    return args


class _HelpAllAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if hasattr(parser, '_unsuppress_hidden'):
            parser._unsuppress_hidden()
        super().__call__(parser, namespace, values, option_string=option_string)


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Produce a custom-formatted `--help` option.

    See https://goo.gl/DKtHb5 for details.
    """

    def __init__(self, *args, **kwargs):
        if 'max_help_position' not in kwargs:
            kwargs['max_help_position'] = 8
        super().__init__(*args, **kwargs)

    def _fill_text(self, text, width, indent):
        # used to ensure that argparse doesn't word-wrap our descriptions of
        # commands. mostly useful for the logo in the supercommand.
        return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _iter_indented_subactions(self, action):
        # used in superscript parser to hide "hidden" commands.
        retval = super()._iter_indented_subactions(action)
        if isinstance(action, argparse._SubParsersAction):
            retval = [x for x in retval if x.help != argparse.SUPPRESS]
        return retval

    def _format_action_invocation(self, action):
        # used to suppress one utterance in the super parser.
        if isinstance(action, argparse._SubParsersAction):
            return ""
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

    def _get_help_string(self, action):
        """
        Help string that (almost) always inserts %(default)s.
        """
        help = action.help
        if (
            '%(default)' in action.help
            or not isinstance(action, argparse._StoreAction)
            or action.default is argparse.SUPPRESS
        ):
            return help

        defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
        if action.option_strings or action.nargs in defaulting_nargs:
            help += ' (default: %(default)s)'
        if (
            hasattr(action, 'recommended')
            and action.recommended
            and action.recommended != action.default
        ):
            help += '(recommended: %(recommended)s)'
            help = help.replace(')(recommended', ', recommended')
        return help


class ParlaiParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI argument parser.

    Pseudo-extension of ``argparse`` which sets a number of parameters
    for the ParlAI framework. More options can be added specific to other
    modules by passing this object and calling ``add_arg()`` or
    ``add_argument()`` on it.

    For an example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.

    :param add_parlai_args:
        (default True) initializes the default arguments for ParlAI
        package, including the data download paths and task arguments.
    :param add_model_args:
        (default False) initializes the default arguments for loading
        models, including initializing arguments from that model.
    """

    def __init__(
        self, add_parlai_args=True, add_model_args=False, description=None, **kwargs
    ):
        """
        Initialize the ParlAI parser.
        """
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = CustomHelpFormatter

        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            add_help=True,
            **kwargs,
        )
        self.register('action', 'helpall', _HelpAllAction)
        self.register('type', 'nonestr', str2none)
        self.register('type', 'bool', str2bool)
        self.register('type', 'floats', str2floats)
        self.register('type', 'multitask_weights', str2multitask_weights)
        self.register('type', 'class', str2class)
        self.parlai_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['PARLAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        # remember which args were specified on the command line
        self.overridable = {}

        if add_parlai_args:
            self.add_parlai_args()
        if add_model_args:
            self.add_model_args()

    def add_parlai_data_path(self, argument_group=None):
        """
        Add --datapath CLI arg.
        """
        if argument_group is None:
            argument_group = self
        argument_group.add_argument(
            '-dp',
            '--datapath',
            default=None,
            help='path to datasets, defaults to {parlai_dir}/data',
        )

    def add_mturk_args(self):
        """
        Add standard mechanical turk arguments.
        """
        mturk = self.add_argument_group('Mechanical Turk')
        default_log_path = os.path.join(self.parlai_home, 'logs', 'mturk')
        mturk.add_argument(
            '--mturk-log-path',
            default=default_log_path,
            help='path to MTurk logs, defaults to {parlai_dir}/logs/mturk',
        )
        mturk.add_argument(
            '-t',
            '--task',
            help='MTurk task, e.g. "qa_data_collection" or "model_evaluator"',
        )
        mturk.add_argument(
            '-nc',
            '--num-conversations',
            default=1,
            type=int,
            help='number of conversations you want to create for this task',
        )
        mturk.add_argument(
            '--unique',
            dest='unique_worker',
            default=False,
            action='store_true',
            help='enforce that no worker can work on your task twice',
        )
        mturk.add_argument(
            '--max-hits-per-worker',
            dest='max_hits_per_worker',
            default=0,
            type=int,
            help='Max number of hits each worker can perform during current group run',
        )
        mturk.add_argument(
            '--unique-qual-name',
            dest='unique_qual_name',
            default=None,
            type=str,
            help='qualification name to use for uniqueness between HITs',
        )
        mturk.add_argument(
            '-r',
            '--reward',
            default=0.05,
            type=float,
            help='reward for each worker for finishing the conversation, '
            'in US dollars',
        )
        mturk.add_argument(
            '--sandbox',
            dest='is_sandbox',
            action='store_true',
            help='submit the HITs to MTurk sandbox site',
        )
        mturk.add_argument(
            '--live',
            dest='is_sandbox',
            action='store_false',
            help='submit the HITs to MTurk live site',
        )
        mturk.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='print and log all server interactions and messages',
        )
        mturk.add_argument(
            '--verbose',
            dest='verbose',
            action='store_true',
            help='print all messages sent to and from Turkers',
        )
        mturk.add_argument(
            '--hard-block',
            dest='hard_block',
            action='store_true',
            default=False,
            help='Hard block disconnecting Turkers from all of your HITs',
        )
        mturk.add_argument(
            '--log-level',
            dest='log_level',
            type=int,
            default=20,
            help='importance level for what to put into the logs. the lower '
            'the level the more that gets logged. values are 0-50',
        )
        mturk.add_argument(
            '--disconnect-qualification',
            dest='disconnect_qualification',
            default=None,
            help='Qualification to use for soft blocking users for '
            'disconnects. By default '
            'turkers are never blocked, though setting this will allow '
            'you to filter out turkers that have disconnected too many '
            'times on previous HITs where this qualification was set.',
        )
        mturk.add_argument(
            '--block-qualification',
            dest='block_qualification',
            default=None,
            help='Qualification to use for soft blocking users. This '
            'qualification is granted whenever soft_block_worker is '
            'called, and can thus be used to filter workers out from a '
            'single task or group of tasks by noted performance.',
        )
        mturk.add_argument(
            '--count-complete',
            dest='count_complete',
            default=False,
            action='store_true',
            help='continue until the requested number of conversations are '
            'completed rather than attempted',
        )
        mturk.add_argument(
            '--allowed-conversations',
            dest='allowed_conversations',
            default=0,
            type=int,
            help='number of concurrent conversations that one mturk worker '
            'is able to be involved in, 0 is unlimited',
        )
        mturk.add_argument(
            '--max-connections',
            dest='max_connections',
            default=30,
            type=int,
            help='number of HITs that can be launched at the same time, 0 is '
            'unlimited.',
        )
        mturk.add_argument(
            '--min-messages',
            dest='min_messages',
            default=0,
            type=int,
            help='number of messages required to be sent by MTurk agent when '
            'considering whether to approve a HIT in the event of a '
            'partner disconnect. I.e. if the number of messages '
            'exceeds this number, the turker can submit the HIT.',
        )
        mturk.add_argument(
            '--local',
            dest='local',
            default=False,
            action='store_true',
            help='Run the server locally on this server rather than setting up'
            ' a heroku server.',
        )
        mturk.add_argument(
            '--hobby',
            dest='hobby',
            default=False,
            action='store_true',
            help='Run the heroku server on the hobby tier.',
        )
        mturk.add_argument(
            '--max-time',
            dest='max_time',
            default=0,
            type=int,
            help='Maximum number of seconds per day that a worker is allowed '
            'to work on this assignment',
        )
        mturk.add_argument(
            '--max-time-qual',
            dest='max_time_qual',
            default=None,
            help='Qualification to use to share the maximum time requirement '
            'with other runs from other machines.',
        )
        mturk.add_argument(
            '--heroku-team',
            dest='heroku_team',
            default=None,
            help='Specify Heroku team name to use for launching Dynos.',
        )
        mturk.add_argument(
            '--tmp-dir',
            dest='tmp_dir',
            default=None,
            help='Specify location to use for scratch builds and such.',
        )

        # it helps to indicate to agents that they're in interactive mode, and
        # can avoid some otherwise troublesome behavior (not having candidates,
        # sharing self.replies, etc).
        mturk.set_defaults(interactive_mode=True)

        mturk.set_defaults(is_sandbox=True)
        mturk.set_defaults(is_debug=False)
        mturk.set_defaults(verbose=False)

    def add_chatservice_args(self):
        """
        Arguments for all chat services.
        """
        args = self.add_argument_group('Chat Services')
        args.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='print and log all server interactions and messages',
        )
        args.add_argument(
            '--config-path',
            default=None,
            type=str,
            help='/path/to/config/file for a given task.',
        )
        args.add_argument(
            '--password',
            dest='password',
            type=str,
            default=None,
            help='Require a password for entry to the bot',
        )

    def add_websockets_args(self):
        """
        Add websocket arguments.
        """
        self.add_chatservice_args()
        websockets = self.add_argument_group('Websockets')
        websockets.add_argument(
            '--port', default=35496, type=int, help='Port to run the websocket handler'
        )

    def add_messenger_args(self):
        """
        Add Facebook Messenger arguments.
        """
        self.add_chatservice_args()
        messenger = self.add_argument_group('Facebook Messenger')
        messenger.add_argument(
            '--verbose',
            dest='verbose',
            action='store_true',
            help='print all messages sent to and from Turkers',
        )
        messenger.add_argument(
            '--log-level',
            dest='log_level',
            type=int,
            default=20,
            help='importance level for what to put into the logs. the lower '
            'the level the more that gets logged. values are 0-50',
        )
        messenger.add_argument(
            '--force-page-token',
            dest='force_page_token',
            action='store_true',
            help='override the page token stored in the cache for a new one',
        )
        messenger.add_argument(
            '--bypass-server-setup',
            dest='bypass_server_setup',
            action='store_true',
            default=False,
            help='should bypass traditional server and socket setup',
        )
        messenger.add_argument(
            '--local',
            dest='local',
            action='store_true',
            default=False,
            help='Run the server locally on this server rather than setting up'
            ' a heroku server.',
        )

        messenger.set_defaults(is_debug=False)
        messenger.set_defaults(verbose=False)

    def add_parlai_args(self, args=None):
        """
        Add common ParlAI args across all scripts.
        """
        self.add_argument(
            '--helpall',
            action='helpall',
            help='Show usage, including advanced arguments.',
        )
        parlai = self.add_argument_group('Main ParlAI Arguments')
        parlai.add_argument(
            '-o',
            '--init-opt',
            default=None,
            help='Path to json file of options. '
            'Note: Further Command-line arguments override file-based options.',
        )
        parlai.add_argument(
            '--allow-missing-init-opts',
            type='bool',
            default=False,
            help=(
                'Warn instead of raising if an argument passed in with --init-opt is '
                'not in the target opt.'
            ),
        )
        parlai.add_argument(
            '-t', '--task', help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"'
        )
        parlai.add_argument(
            '--download-path',
            default=None,
            hidden=True,
            help='path for non-data dependencies to store any needed files.'
            'defaults to {parlai_dir}/downloads',
        )
        parlai.add_argument(
            '--loglevel',
            default='info',
            hidden=True,
            choices=get_all_levels(),
            help='Logging level',
        )
        parlai.add_argument(
            '-dt',
            '--datatype',
            metavar='DATATYPE',
            default='train',
            choices=[
                'train',
                'train:stream',
                'train:ordered',
                'train:ordered:stream',
                'train:stream:ordered',
                'train:evalmode',
                'train:evalmode:stream',
                'train:evalmode:ordered',
                'train:evalmode:ordered:stream',
                'train:evalmode:stream:ordered',
                'valid',
                'valid:stream',
                'test',
                'test:stream',
            ],
            help='choose from: train, train:ordered, valid, test. to stream '
            'data add ":stream" to any option (e.g., train:stream). '
            'by default train is random with replacement, '
            'valid is ordered, test is ordered.',
        )
        parlai.add_argument(
            '-im',
            '--image-mode',
            default='raw',
            type=str,
            help='image preprocessor to use. default is "raw". set to "none" '
            'to skip image loading.',
            hidden=True,
        )
        parlai.add_argument(
            '--hide-labels',
            default=False,
            type='bool',
            hidden=True,
            help='default (False) moves labels in valid and test sets to the '
            'eval_labels field. If True, they are hidden completely.',
        )
        parlai.add_argument(
            '-mtw',
            '--multitask-weights',
            type='multitask_weights',
            default=[1],
            help=(
                'list of floats, one for each task, specifying '
                'the probability of drawing the task in multitask case. You may also '
                'provide "stochastic" to simulate simple concatenation.'
            ),
            hidden=True,
        )
        parlai.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
        )
        parlai.add_argument(
            '-dynb',
            '--dynamic-batching',
            default=None,
            type='nonestr',
            choices={None, 'full', 'batchsort'},
            help='Use dynamic batching',
        )
        parlai.add_argument(
            '-v',
            '--verbose',
            dest='verbose',
            action='store_true',
            help='Print all messages',
        )
        parlai.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='Enables some debug behavior',
        )
        self.add_parlai_data_path(parlai)

    def add_distributed_training_args(self):
        """
        Add CLI args for distributed training.
        """
        grp = self.add_argument_group('Distributed Training')
        grp.add_argument(
            '--distributed-world-size', type=int, help='Number of workers.'
        )
        grp.add_argument(
            '--ddp-backend',
            choices=['ddp', 'zero2', 'zero3'],
            default='ddp',
            help=(
                'Distributed backend. Zero2 can be faster but is more experimental. '
                'Zero3 significantly reduces memory pressure. '
                'DDP is the most tested.'
            ),
        )
        return grp

    def add_model_args(self):
        """
        Add arguments related to models such as model files.
        """
        model_args = self.add_argument_group('ParlAI Model Arguments')
        model_args.add_argument(
            '-m',
            '--model',
            default=None,
            help='the model class name. can match parlai/agents/<model> for '
            'agents in that directory, or can provide a fully specified '
            'module for `from X import Y` via `-m X:Y` '
            '(e.g. `-m parlai.agents.seq2seq.seq2seq:Seq2SeqAgent`)',
        )
        model_args.add_argument(
            '-mf',
            '--model-file',
            default=None,
            help='model file name for loading and saving models',
        )
        model_args.add_argument(
            '-im',
            '--init-model',
            default=None,
            type=str,
            help='Initialize model weights and dict from this file',
        )
        model_args.add_argument(
            '--dict-class', hidden=True, help='the class of the dictionary agent uses'
        )

    def add_model_subargs(self, model: str, partial: Opt):
        """
        Add arguments specific to a particular model.
        """
        agent = load_agent_module(model)
        try:
            if hasattr(agent, 'add_cmdline_args'):
                agent.add_cmdline_args(self, partial)
        except TypeError as typ:
            raise TypeError(
                f"Agent '{model}' appears to have signature "
                "add_cmdline_args(argparser) but we have updated the signature "
                "to add_cmdline_args(argparser, partial_opt). For details, see "
                "https://github.com/facebookresearch/ParlAI/pull/3328."
            ) from typ
        except argparse.ArgumentError:
            # already added
            pass
        try:
            if hasattr(agent, 'dictionary_class'):
                s = class2str(agent.dictionary_class())
                self.set_defaults(dict_class=s)
        except argparse.ArgumentError:
            # already added
            pass

    def add_task_args(self, task: str, partial: Opt):
        """
        Add arguments specific to the specified task.
        """
        for t in ids_to_tasks(task).split(','):
            agent = load_teacher_module(t)
            try:
                if hasattr(agent, 'add_cmdline_args'):
                    agent.add_cmdline_args(self, partial)
            except TypeError as typ:
                raise TypeError(
                    f"Task '{task}' appears to have signature "
                    "add_cmdline_args(argparser) but we have updated the signature "
                    "to add_cmdline_args(argparser, partial_opt). For details, see "
                    "https://github.com/facebookresearch/ParlAI/pull/3328."
                ) from typ
            except argparse.ArgumentError:
                # already added
                pass

    def add_world_args(
        self,
        task: str,
        interactive_task: Optional[str],
        selfchat_task: Optional[str],
        partial: Opt,
    ):
        """
        Add arguments specific to the world.
        """
        world_class = load_world_module(
            task, interactive_task=interactive_task, selfchat_task=selfchat_task
        )
        if world_class is not None and hasattr(world_class, 'add_cmdline_args'):
            try:
                world_class.add_cmdline_args(self, partial)
            except argparse.ArgumentError:
                # already added
                pass
            except TypeError:
                raise TypeError(
                    f"World '{task}' appears to have signature "
                    "add_cmdline_args(argparser) but we have updated the signature "
                    "to add_cmdline_args(argparser, partial_opt). For details, see "
                    "https://github.com/facebookresearch/ParlAI/pull/3328."
                )

    def add_image_args(self, image_mode):
        """
        Add additional arguments for handling images.
        """
        try:
            parlai = self.add_argument_group('ParlAI Image Preprocessing Arguments')
            parlai.add_argument(
                '--image-size',
                type=int,
                default=256,
                help='resizing dimension for images',
                hidden=True,
            )
            parlai.add_argument(
                '--image-cropsize',
                type=int,
                default=224,
                help='crop dimension for images',
                hidden=True,
            )
        except argparse.ArgumentError:
            # already added
            pass

    def add_extra_args(self, args=None):
        """
        Add more args depending on how known args are set.
        """
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])
        # Also load extra args options if a file is given.
        if parsed.get('init_opt') is not None:
            try:
                self._load_known_opts(parsed.get('init_opt'), parsed)
            except FileNotFoundError:
                # don't die if -o isn't found here. See comment in second call
                # later on.
                pass
        parsed = self._infer_datapath(parsed)

        partial = Opt(parsed)

        # find which image mode specified if any, and add additional arguments
        image_mode = parsed.get('image_mode', None)
        if image_mode is not None and image_mode != 'no_image_model':
            self.add_image_args(image_mode)

        # find which task specified if any, and add its specific arguments
        task = parsed.get('task', None)
        if task is not None:
            self.add_task_args(task, partial)
        evaltask = parsed.get('evaltask', None)
        if evaltask is not None:
            self.add_task_args(evaltask, partial)

        # find which model specified if any, and add its specific arguments
        model = get_model_name(parsed)
        if model is not None:
            self.add_model_subargs(model, partial)

        # add world args, if we know a priori which world is being used
        if task is not None:
            self.add_world_args(
                task,
                parsed.get('interactive_task', False),
                parsed.get('selfchat_task', False),
                partial,
            )

        # reparse args now that we've inferred some things.  specifically helps
        # with a misparse of `-opt` as `-o pt`, which causes opt loading to
        # try to load the file "pt" which doesn't exist.
        # After adding model arguments, -opt becomes known (it's in TorchAgent),
        # and we parse the `-opt` value correctly.
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])
        if parsed.get('init_opt') is not None:
            self._load_known_opts(parsed.get('init_opt'), parsed)

        # reset parser-level defaults over any model-level defaults
        try:
            self.set_defaults(**self._defaults)
        except AttributeError:
            raise RuntimeError(
                'Please file an issue on github that argparse '
                'got an attribute error when parsing.'
            )

    def _handle_single_dash_parsearg(self, args, actions):
        if _sys.version_info >= (3, 8, 0):
            newargs = []
            for arg in args:
                darg = f'-{arg}'
                if arg.startswith('-') and not arg.startswith('--') and darg in actions:
                    newargs.append(darg)
                else:
                    newargs.append(arg)
            return newargs
        else:
            return args

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        """
        Parse known args to ignore help flag.
        """
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]

        args = fix_underscores(args)
        # handle the single dash stuff. See _handle_single_dash_addarg for info
        actions = set()
        for action in self._actions:
            actions.update(action.option_strings)
        args = self._handle_single_dash_parsearg(args, actions)
        if nohelp:
            # ignore help
            args = [
                a
                for a in args
                if a != '-h' and a != '--help' and a != '--helpall' and a != '--h'
            ]
        return super().parse_known_args(args, namespace)

    def _load_known_opts(self, optfile, parsed):
        """
        Pull in CLI args for proper models/tasks/etc.

        Called before args are parsed; ``_load_opts`` is used for actually overriding
        opts after they are parsed.
        """
        new_opt = Opt.load_init(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in parsed or parsed[key] is None:
                parsed[key] = value

    def _load_opts(self, opt):
        optfile = opt.get('init_opt')
        new_opt = Opt.load_init(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in opt:
                if opt.get('allow_missing_init_opts', False):
                    logging.warning(
                        f'The "{key}" key in {optfile} will not be loaded, because it '
                        f'does not exist in the target opt.'
                    )
                else:
                    raise RuntimeError(
                        'Trying to set opt from file that does not exist: ' + str(key)
                    )
            if key not in opt['override']:
                opt[key] = value
                opt['override'][key] = value

    def _infer_datapath(self, opt):
        """
        Set the value for opt['datapath'] and opt['download_path'].

        Sets the value for opt['datapath'] and opt['download_path'], correctly
        respecting environmental variables and the default.
        """
        # set environment variables
        # Priority for setting the datapath (same applies for download_path):
        # --datapath -> os.environ['PARLAI_DATAPATH'] -> <self.parlai_home>/data
        if opt.get('datapath'):
            os.environ['PARLAI_DATAPATH'] = opt['datapath']
        elif os.environ.get('PARLAI_DATAPATH') is None:
            DEFAULT_DATAPATH = None
            try:
                # internal fbcode-wide default
                import parlai_fb

                DEFAULT_DATAPATH = parlai_fb.DEFAULT_DATAPATH
            except ImportError:
                pass
            if not DEFAULT_DATAPATH:
                # TODO: switch to ~/.parlai/
                DEFAULT_DATAPATH = os.path.join(self.parlai_home, 'data')
            os.environ['PARLAI_DATAPATH'] = DEFAULT_DATAPATH

        opt['datapath'] = os.environ['PARLAI_DATAPATH']

        return opt

    def _process_args_to_opts(self, args_that_override: Optional[List[str]] = None):
        self.opt = Opt(vars(self.args))
        extra_ag = []

        if '_subparser' in self.opt:
            # if using the super command, we need to be aware of the subcommand's
            # arguments when identifying things manually set by the user
            self.overridable.update(self.opt['_subparser'].overridable)
            extra_ag = self.opt.pop('_subparser')._action_groups

        # custom post-parsing
        self.opt['parlai_home'] = self.parlai_home
        self.opt = self._infer_datapath(self.opt)

        # set all arguments specified in command line as overridable
        option_strings_dict = {}
        store_true = []
        store_false = []
        for group in self._action_groups + extra_ag:
            for a in group._group_actions:
                if hasattr(a, 'option_strings'):
                    for option in a.option_strings:
                        option_strings_dict[option] = a.dest
                        if isinstance(a, argparse._StoreTrueAction):
                            store_true.append(option)
                        elif isinstance(a, argparse._StoreFalseAction):
                            store_false.append(option)

        if args_that_override is None:
            args_that_override = _sys.argv[1:]

        args_that_override = self._handle_single_dash_parsearg(
            fix_underscores(args_that_override), option_strings_dict.keys()
        )

        for i in range(len(args_that_override)):
            if args_that_override[i] in option_strings_dict:
                if args_that_override[i] in store_true:
                    self.overridable[option_strings_dict[args_that_override[i]]] = True
                elif args_that_override[i] in store_false:
                    self.overridable[option_strings_dict[args_that_override[i]]] = False
                elif (
                    i < len(args_that_override) - 1
                    and args_that_override[i + 1] not in option_strings_dict
                ):
                    key = option_strings_dict[args_that_override[i]]
                    self.overridable[key] = self.opt[key]
        self.opt['override'] = self.overridable

        # load opts if a file is provided.
        if self.opt.get('init_opt', None) is not None:
            self._load_opts(self.opt)

        # map filenames that start with 'zoo:' to point to the model zoo dir
        options_to_change = {'model_file', 'dict_file', 'bpe_vocab', 'bpe_merge'}
        for each_key in options_to_change:
            if self.opt.get(each_key) is not None:
                self.opt[each_key] = modelzoo_path(
                    self.opt.get('datapath'), self.opt[each_key]
                )
            if self.opt['override'].get(each_key) is not None:
                # also check override
                self.opt['override'][each_key] = modelzoo_path(
                    self.opt.get('datapath'), self.opt['override'][each_key]
                )

        # add start time of an experiment
        self.opt['starttime'] = datetime.datetime.today().strftime('%b%d_%H-%M')

    def parse_and_process_known_args(self, args=None):
        """
        Parse provided arguments and return parlai opts and unknown arg list.

        Runs the same arg->opt parsing that parse_args does, but doesn't throw an error
        if the args being parsed include additional command line arguments that parlai
        doesn't know what to do with.
        """
        self.args, unknowns = super().parse_known_args(args=args)
        self._process_args_to_opts(args)
        return self.opt, unknowns

    def parse_args(self, args=None, namespace=None, **kwargs):
        """
        Parse the provided arguments and returns a dictionary of the ``args``.

        We specifically remove items with ``None`` as values in order to support the
        style ``opt.get(key, default)``, which would otherwise return ``None``.
        """
        if 'print_args' in kwargs:
            logging.error(
                "You gave the print_args flag to parser.parse_args, but this is "
                "no longer supported. Use opt.log() to print the arguments"
            )
            del kwargs['print_args']
        self.add_extra_args(args)
        self.args = super().parse_args(args=args)

        self._process_args_to_opts(args)
        print_announcements(self.opt)

        assert '_subparser' not in self.opt

        return self.opt

    def _value2argstr(self, value) -> str:
        """
        Reverse-parse an opt value into one interpretable by argparse.
        """
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        else:
            return str(value)

    def _kwargs_to_str_args(self, **kwargs):
        """
        Attempt to map from python-code kwargs into CLI args.

        e.g. model_file -> --model-file.

        Works with short options too, like t="convai2".
        """

        # we have to do this large block of repetitive code twice, the first
        # round is basically just to become aware of anything that would have
        # been added by add_extra_args
        kwname_to_action = {}
        for action in self._actions:
            if action.dest == 'help':
                # no help allowed
                continue
            for option_string in action.option_strings:
                kwname = option_string.lstrip('-').replace('-', '_')
                assert (kwname not in kwname_to_action) or (
                    kwname_to_action[kwname] is action
                ), f"No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})"
                kwname_to_action[kwname] = action

        # since we can have options that depend on options, repeat until convergence
        string_args = []
        unparsed_args = set(kwargs.keys())
        while unparsed_args:
            string_args = []
            for kwname, value in kwargs.items():
                if kwname not in kwname_to_action:
                    # best guess, we need to delay it. hopefully this gets added
                    # during add_kw_Args
                    continue
                action = kwname_to_action[kwname]
                last_option_string = action.option_strings[-1]
                if isinstance(action, argparse._StoreTrueAction):
                    if bool(value):
                        string_args.append(last_option_string)
                elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                    string_args.append(last_option_string)
                    string_args.append(self._value2argstr(value))
                elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                    string_args.append(last_option_string)
                    string_args.extend([self._value2argstr(v) for v in value])
                else:
                    raise TypeError(f"Don't know what to do with {action}")

            # become aware of any extra args that might be specified if the user
            # provides something like model="transformer/generator".
            self.add_extra_args(string_args)

            # do it again, this time knowing about ALL args.
            kwname_to_action = {}
            for action in self._actions:
                if action.dest == 'help':
                    # no help allowed
                    continue
                for option_string in action.option_strings:
                    kwname = option_string.lstrip('-').replace('-', '_')
                    assert (kwname not in kwname_to_action) or (
                        kwname_to_action[kwname] is action
                    ), f"No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})"
                    kwname_to_action[kwname] = action

            new_unparsed_args = set()
            string_args = []
            for kwname, value in kwargs.items():
                if kwname not in kwname_to_action:
                    new_unparsed_args.add(kwname)
                    continue

                action = kwname_to_action[kwname]
                last_option_string = action.option_strings[-1]
                if isinstance(action, argparse._StoreTrueAction):
                    if bool(value):
                        string_args.append(last_option_string)
                elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                    string_args.append(last_option_string)
                    string_args.append(self._value2argstr(value))
                elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                    string_args.append(last_option_string)
                    # Special case: Labels
                    string_args.extend([self._value2argstr(v) for v in value])
                else:
                    raise TypeError(f"Don't know what to do with {action}")

            if new_unparsed_args == unparsed_args:
                # if we have converged to a fixed point with no improvements, we
                # truly found some unreachable args
                raise KeyError(
                    f'Failed to parse one or more kwargs: {", ".join(new_unparsed_args)}'
                )
            else:
                # We've seen some improvements on the number of unparsed args,
                # iterate again
                unparsed_args = new_unparsed_args

        return string_args

    def parse_kwargs(self, **kwargs):
        """
        Parse kwargs, with type checking etc.
        """
        # hack: capture any error messages without raising a SystemExit
        def _captured_error(msg):
            raise ValueError(msg)

        old_error = self.error
        self.error = _captured_error
        try:
            string_args = self._kwargs_to_str_args(**kwargs)
            return self.parse_args(args=string_args)
        finally:
            self.error = old_error

    def set_params(self, **kwargs):
        """
        Set overridable kwargs.
        """
        self.set_defaults(**kwargs)
        for k, v in kwargs.items():
            self.overridable[k] = v

    def _unsuppress_hidden(self):
        for action in self._actions:
            if hasattr(action, 'real_help'):
                action.help = action.real_help

    def _handle_custom_options(self, kwargs):
        """
        Handle custom parlai options.

        Includes hidden, recommended. Future may include no_save and no_override.
        """
        action_attr = {}
        if 'recommended' in kwargs:
            rec = kwargs.pop('recommended')
            action_attr['recommended'] = rec
        action_attr['hidden'] = kwargs.get('hidden', False)
        action_attr['real_help'] = kwargs.get('help', None)
        if 'hidden' in kwargs:
            if kwargs.pop('hidden'):
                kwargs['help'] = argparse.SUPPRESS

        if 'type' in kwargs and kwargs['type'] is bool:
            # common error, we really want simple form
            kwargs['type'] = 'bool'
        return kwargs, action_attr

    def _handle_single_dash_addarg(self, args):
        """
        Fixup argparse for parlai-style short args.

        In python 3.8, argparsing was changed such that short arguments are not
        required to have spaces after them. This causes our many short args to
        be misinterpetted by the parser. For example `-emb` gets parsed as
        `-e mb`.

        Here we rewrite them into long args to get around the nonsense.
        """
        if _sys.version_info < (3, 8, 0):
            # older python works fine
            return args

        # need the long options specified first, or `dest` will get set to
        # the short name on accident!
        out_long = []
        out_short = []
        for arg in args:
            if arg.startswith('-') and not arg.startswith('--'):
                out_short.append(f'-{arg}')
            else:
                out_long.append(arg)
        # keep long args in front so they are used for the destination
        return out_long + out_short

    def add_argument(self, *args, **kwargs):
        """
        Override to convert underscores to hyphens for consistency.
        """
        kwargs, newattr = self._handle_custom_options(kwargs)
        args = self._handle_single_dash_addarg(fix_underscores(args))
        action = super().add_argument(*args, **kwargs)
        for k, v in newattr.items():
            setattr(action, k, v)
        return action

    def add_argument_group(self, *args, **kwargs):
        """
        Override to make arg groups also convert underscores to hyphens.
        """
        arg_group = super().add_argument_group(*args, **kwargs)
        original_add_arg = arg_group.add_argument

        def ag_add_argument(*args, **kwargs):
            kwargs, newattr = self._handle_custom_options(kwargs)
            args = self._handle_single_dash_addarg(fix_underscores(args))
            action = original_add_arg(*args, **kwargs)
            for k, v in newattr.items():
                setattr(action, k, v)
            return action

        arg_group.add_argument = ag_add_argument  # override _ => -
        arg_group.add_argument_group = self.add_argument_group
        return arg_group

    def error(self, message):
        """
        Override to print custom error message.
        """
        self.print_help()
        _sys.stderr.write('\nParse Error: %s\n' % message)
        _sys.exit(2)


def default(val, default):
    """
    shorthand for explicit None check for optional arguments.
    """
    return val if val is not None else default


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'Interactive chat with a model on the command line'
        )
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-add-fields',
        type=str,
        default='',
        help='Display these fields when verbose is off (e.g., "--display-add-fields label_candidates,beam_texts")',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default='',
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Set to the empty string to not save at all',
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    # LocalHumanAgent.add_cmdline_args(parser, partial_opt=None)
    agent = parser.add_argument_group('Local Human Arguments')
    agent.add_argument(
            '-fixedCands',
            '--local-human-candidates-file',
            default=None,
            type=str,
            help='File of label_candidates to send to other agent',
        )
    agent.add_argument(
            '--single_turn',
            type='bool',
            default=False,
            help='If on, assumes single turn episodes.',
        )
    # WorldLogger.add_cmdline_args(parser, partial_opt=None)
    agent = parser.add_argument_group('World Logging')
    agent.add_argument(
            '--log-keep-fields',
            type=str,
            default=KEEP_ALL,
            help='Fields to keep when logging. Should be a comma separated list',
        )
    return parser


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for downloading and building data.

These can be replaced if your particular file system does not support them.
"""

import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import gzip
import math
import contextlib
# import parlai.utils.logging as logging
# from parlai.utils.io import PathManager

try:
    from torch.multiprocessing import Pool
except ImportError:
    from multiprocessing import Pool


try:
    # internal infra requires special attention to use http sessions
    from parlai_fb import get_http_session
except (ImportError, AttributeError):

    @contextlib.contextmanager
    def get_http_session():
        with requests.Session() as session:
            yield session


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Any task that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Untar the file if zipped
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link

    An object of this class needs to be created with:

    - url <string> : URL or Google Drive id to download from
    - file_name <string> : File name that the file should be named
    - hashcode <string> : SHA256 hashcode of the downloaded file
    - zipped <boolean> : False if the file is not compressed
    - from_google <boolean> : True if the file is from Google Drive
    """

    def __init__(self, url, file_name, hashcode, zipped=True, from_google=False):
        self.url = url
        self.file_name = file_name
        self.hashcode = hashcode
        self.zipped = zipped
        self.from_google = from_google

    def checksum(self, dpath):
        """
        Checksum on a given file.

        :param dpath: path to the downloaded file.
        """
        sha256_hash = hashlib.sha256()
        with PathManager.open(os.path.join(dpath, self.file_name), "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self.hashcode:
                # remove_dir(dpath)
                raise AssertionError(
                    f"Checksum for {self.file_name} from \n{self.url}\n"
                    f"does not match the expected checksum:\n"
                    f"{sha256_hash.hexdigest()} (received) != {self.hashcode} (expected)\n"
                    f"\nPlease try again. You may need to manually delete {self.file_name}."
                )
            else:
                logging.debug("Checksum Successful")

    def download_file(self, dpath):
        if self.from_google:
            download_from_google_drive(self.url, os.path.join(dpath, self.file_name))
        else:
            download(self.url, dpath, self.file_name)

        self.checksum(dpath)

        if self.zipped:
            untar(dpath, self.file_name)

    def check_header(self):
        """
        Performs a HEAD request to check if the URL / Google Drive ID is live.
        """
        with get_http_session() as session:
            if self.from_google:
                URL = 'https://docs.google.com/uc?export=download'
                response = session.head(URL, params={'id': self.url}, stream=True)
            else:
                headers = {
                    'User-Agent': (
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/77.0.3865.90 Safari/537.36'
                    )
                }
                response = session.head(self.url, allow_redirects=True, headers=headers)
            status = response.status_code

        assert status == 200


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version is regarded as
    not built.
    """
    if version_string:
        fname = os.path.join(path, '.built')
        if not PathManager.exists(fname):
            return False
        else:
            with PathManager.open(fname, 'r') as read:
                text = read.read().split('\n')
            return len(text) > 1 and text[1] == version_string
    else:
        return PathManager.exists(os.path.join(path, '.built'))


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    :param str path:
        The file path to mark as built.

    :param str version_string:
        The version of this dataset.
    """
    with PathManager.open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)


def download(url, path, fname, redownload=False, num_retries=5):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``False``).
    """
    outfile = os.path.join(path, fname)
    download = not PathManager.exists(outfile) or redownload
    logging.info(f"Downloading {url} to {outfile}")
    retry = num_retries
    exp_backoff = [2**r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(fname))

    while download and retry > 0:
        response = None

        with get_http_session() as session:
            try:
                response = session.get(url, stream=True, timeout=5)

                # negative reply could be 'none' or just missing
                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                pbar.total = total_size
                done = 0

                with PathManager.open(outfile, 'wb') as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry > 0:
                    pl = 'y' if retry == 1 else 'ies'
                    logging.debug(
                        f'Connection error, retrying. ({retry} retr{pl} left)'
                    )
                    time.sleep(exp_backoff[retry])
                else:
                    logging.error('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry <= 0:
        raise RuntimeError('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeError(
                f'Received less data than specified in Content-Length header for '
                f'{url}. There may be a download problem.'
            )

    pbar.close()


def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    # the current working directory is a fine path
    if path != '':
        PathManager.mkdirs(path)


def remove_dir(path):
    """
    Remove the given directory, if it exists.
    """
    shutil.rmtree(path, ignore_errors=True)


def untar(path, fname, delete=True, flatten_tar=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    if ".zip" in fname:
        return _unzip(path, fname, delete=delete)
    else:
        return _untar(path, fname, delete=delete, flatten=flatten_tar)


def _untar(path, fname, delete=True, flatten=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import tarfile

    logging.debug(f'unpacking {fname}')
    fullpath = os.path.join(path, fname)
    # very painfully manually extract files so that we can use PathManger.open
    # instead, lest we are using fb internal file services

    with tarfile.open(fileobj=PathManager.open(fullpath, 'rb')) as tf:
        for item in tf:
            item_name = item.name
            while item_name.startswith("./"):
                # internal file systems will actually create a literal "."
                # directory, so we gotta watch out for that
                item_name = item_name[2:]
            if flatten:
                # flatten the tar file if there are subdirectories
                fn = os.path.join(path, os.path.split(item_name)[-1])
            else:
                fn = os.path.join(path, item_name)
            logging.debug(f"Extracting to {fn}")
            if item.isdir():
                PathManager.mkdirs(fn)
            elif item.isfile():
                with PathManager.open(fn, 'wb') as wf, tf.extractfile(item.name) as rf:
                    tarfile.copyfileobj(rf, wf)
            else:
                raise NotImplementedError("No support for symlinks etc. right now.")

    if delete:
        try:
            PathManager.rm(fullpath)
        except PermissionError:
            logging.error(
                f"Tried to delete {fullpath} but got a permission error. This "
                "is known to happen in Windows and is probably not fatal."
            )


def ungzip(path, fname, deleteGZip=True):
    """
    Unzips the given gzip compressed file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool deleteGZip:
        If true, the compressed file will be deleted after extraction.
    """

    def _get_output_filename(input_fname):
        GZIP_EXTENSIONS = ('.gz', '.gzip', '.tgz', '.tar')
        for ext in GZIP_EXTENSIONS:
            if input_fname.endswith(ext):
                return input_fname[: -len(ext)]
        return f'{input_fname}_unzip'

    logging.debug(f'unzipping {fname}')
    fullpath = os.path.join(path, fname)

    with gzip.open(PathManager.open(fullpath, 'rb'), 'r') as fin, PathManager.open(
        _get_output_filename(fullpath), 'wb'
    ) as fout:
        shutil.copyfileobj(fin, fout)

    if deleteGZip:
        os.remove(fullpath)


def _unzip(path, fname, delete=True):
    """
    Unpack the given zip file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import zipfile

    logging.debug(f'unpacking {fname}')
    fullpath = os.path.join(path, fname)
    with zipfile.ZipFile(PathManager.open(fullpath, 'rb'), 'r') as zf:
        for member in zf.namelist():
            outpath = os.path.join(path, member)
            if zf.getinfo(member).is_dir():
                logging.debug(f"Making directory {outpath}")
                PathManager.mkdirs(outpath)
                continue
            logging.debug(f"Extracting to {outpath}")
            try:
                with zf.open(member, 'r') as inf, PathManager.open(
                    outpath, 'wb'
                ) as outf:
                    shutil.copyfileobj(inf, outf)
            except FileNotFoundError:
                logging.error(f"Failed to open ${member} and extract to ${outpath}")
    if delete:
        try:
            PathManager.rm(fullpath)
        except PermissionError:
            logging.error(
                f"Tried to delete {fullpath} but got a permission error. This "
                "is known to happen in Windows and is probably not fatal."
            )


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_from_google_drive(gd_id, destination):
    """
    Use the requests package to download a file from Google Drive.
    """
    URL = 'https://docs.google.com/uc?export=download'

    with get_http_session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response) or 't'

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with PathManager.open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()


def get_model_dir(datapath):
    return os.path.join(datapath, 'models')


def download_models(
    opt,
    fnames,
    model_folder,
    version='v1.0',
    path='aws',
    use_model_type=False,
    flatten_tar=False,
):
    """
    Download models into the ParlAI model zoo from a url.

    :param fnames: list of filenames to download
    :param model_folder: models will be downloaded into models/model_folder/model_type
    :param path: url for downloading models; defaults to downloading from AWS
    :param use_model_type: whether models are categorized by type in AWS
    """
    model_type = opt.get('model_type', None)
    if model_type is not None:
        dpath = os.path.join(opt['datapath'], 'models', model_folder, model_type)
    else:
        dpath = os.path.join(opt['datapath'], 'models', model_folder)

    if not built(dpath, version):
        for fname in fnames:
            logging.info(f'building data: {dpath}/{fname}')
        if built(dpath):
            # An older version exists, so remove these outdated files.
            remove_dir(dpath)
        make_dir(dpath)

        # Download the data.
        for fname in fnames:
            if path == 'aws':
                url = 'http://parl.ai/downloads/_models/'
                url += model_folder + '/'
                if use_model_type:
                    url += model_type + '/'
                url += fname
            else:
                url = path + '/' + fname
            download(url, dpath, fname)
            if '.tgz' in fname or '.gz' in fname or '.zip' in fname:
                untar(dpath, fname, flatten_tar=flatten_tar)
        # Mark the data as built.
        mark_done(dpath, version)


def modelzoo_path(datapath, path):
    """
    Map pretrain models filenames to their path on disk.

    If path starts with 'models:', then we remap it to the model zoo path within the
    data directory (default is ParlAI/data/models). We download models from the model
    zoo if they are not here yet.
    """
    if path is None:
        return None
    if (
        not path.startswith('models:')
        and not path.startswith('zoo:')
        and not path.startswith('izoo:')
    ):
        return path
    elif path.startswith('models:') or path.startswith('zoo:'):
        zoo = path.split(':')[0]
        zoo_len = len(zoo) + 1
        model_path = path[zoo_len:]
        # Check if we need to download the model
        if "/" in path:
            animal = path[zoo_len : path.rfind('/')].replace('/', '.')
        else:
            animal = path[zoo_len:]
        if '.' not in animal:
            animal += '.build'
        module_name = 'parlai.zoo.{}'.format(animal)
        try:
            my_module = importlib.import_module(module_name)
            my_module.download(datapath)
        except (ImportError, AttributeError):
            try:
                # maybe we didn't find a specific model, let's try generic .build
                animal_ = '.'.join(animal.split(".")[:-1]) + '.build'
                module_name_ = 'parlai.zoo.{}'.format(animal_)
                my_module = importlib.import_module(module_name_)
                my_module.download(datapath)
            except (ImportError, AttributeError) as exc:
                # truly give up
                raise ImportError(
                    f'Could not find pretrained model in {module_name} or {module_name_}.'
                    ' Please check your spelling and make sure you\'ve pulled from master.'
                ) from exc

        return os.path.join(datapath, 'models', model_path)
    else:
        # Internal path (starts with "izoo:") -- useful for non-public
        # projects.  Save the path to your internal model zoo in
        # parlai_internal/.internal_zoo_path
        # TODO: test the internal zoo.
        zoo_path = 'parlai_internal/zoo/.internal_zoo_path'
        if not PathManager.exists('parlai_internal/zoo/.internal_zoo_path'):
            raise RuntimeError(
                'Please specify the path to your internal zoo in the '
                'file parlai_internal/zoo/.internal_zoo_path in your '
                'internal repository.'
            )
        else:
            with PathManager.open(zoo_path, 'r') as f:
                zoo = f.read().split('\n')[0]
            return os.path.join(zoo, path[5:])


def download_multiprocess(
    urls, path, num_processes=32, chunk_size=100, dest_filenames=None, error_path=None
):
    """
    Download items in parallel (e.g. for an image + dialogue task).

    WARNING: may have issues with OS X.

    :param urls:
        Array of urls to download
    :param path:
        directory to save items in
    :param num_processes:
        number of processes to use
    :param chunk_size:
        chunk size to use
    :param dest_filenames:
        optional array of same length as url with filenames.  Images will be
        saved as path + dest_filename
    :param error_path:
        where to save error logs
    :return:
        array of tuples of (destination filename, http status code, error
        message if any). Note that upon failure, file may not actually be
        created.
    """

    pbar = tqdm.tqdm(total=len(urls), position=0)

    # Resume TODO: isfile() may take too long ?? Should I try in a .tmp file
    if dest_filenames:
        if len(dest_filenames) != len(urls):
            raise Exception(
                'If specified, destination filenames must equal url array in length.'
            )
    else:

        def _naming_fn(url, url_metadata=None):
            return hashlib.md5(url.encode('utf-8')).hexdigest()

        dest_filenames = [_naming_fn(url) for url in urls]

    items = zip(urls, dest_filenames)
    remaining_items = [
        it for it in items if not PathManager.exists(os.path.join(path, it[1]))
    ]
    logging.info(
        f'Of {len(urls)} items, {len(urls) - len(remaining_items)} already existed; only going to download {len(remaining_items)} items.'
    )
    pbar.update(len(urls) - len(remaining_items))

    pool_chunks = (
        (remaining_items[i : i + chunk_size], path, _download_multiprocess_single)
        for i in range(0, len(remaining_items), chunk_size)
    )
    remaining_chunks_count = math.ceil(float(len(remaining_items) / chunk_size))
    logging.info(
        f'Going to download {remaining_chunks_count} chunks with {chunk_size} images per chunk using {num_processes} processes.'
    )

    pbar.desc = 'Downloading'
    all_results = []
    collected_errors = []

    with Pool(num_processes) as pool:
        for idx, chunk_result in enumerate(
            pool.imap_unordered(_download_multiprocess_map_chunk, pool_chunks, 2)
        ):
            all_results.extend(chunk_result)
            for dest_file, http_status_code, error_msg in chunk_result:
                if http_status_code != 200:
                    # msg field available as third item in the tuple
                    # not using b/c error log file would blow up
                    collected_errors.append(
                        {
                            'dest_file': dest_file,
                            'status_code': http_status_code,
                            'error': error_msg,
                        }
                    )
                    logging.error(
                        f'Bad download - chunk: {idx}, dest_file: {dest_file}, http status code: {http_status_code}, error_msg: {error_msg}'
                    )
            pbar.update(len(chunk_result))
    pbar.close()

    if error_path:
        now = time.strftime("%Y%m%d-%H%M%S")
        error_filename = os.path.join(
            error_path, 'parlai_download_multiprocess_errors_%s.log' % now
        )

        with PathManager.open(os.path.join(error_filename), 'w') as error_file:
            error_file.write(json.dumps(collected_errors))
            logging.error(f'Summary of errors written to {error_filename}')

    logging.info(
        f'Of {len(remaining_items)} items attempted downloading, '
        f'{len(collected_errors)} had errors.'
    )

    logging.debug('Finished downloading chunks.')
    return all_results


def _download_multiprocess_map_chunk(pool_tup):
    """
    Helper function for Pool imap_unordered.

    Apparently function must be pickable (which apparently means must be
    defined at the top level of a module and can't be a lamdba) to be used in
    imap_unordered. Has to do with how it's passed to the subprocess.

    :param pool_tup: is a tuple where first arg is an array of tuples of url
    and dest file name for the current chunk and second arg is function to be
    called.
    :return: an array of tuples
    """
    items = pool_tup[0]
    path = pool_tup[1]
    fn = pool_tup[2]
    return [fn(it[0], path, it[1]) for it in items]


def _download_multiprocess_single(url, path, dest_fname):
    """
    Helper function to download an individual item.

    Unlike download() above, does not deal with downloading chunks of a big
    file, does not support retries (and does not fail if retries are exhausted).

    :param url: URL to download from
    :param path: directory to save in
    :param dest_fname: destination file name of image
    :return tuple (dest_fname, http status)
    """

    status = None
    error_msg = None
    try:
        # 'User-Agent' header may need to be specified
        headers = {}

        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(
            url, stream=False, timeout=10, allow_redirects=True, headers=headers
        )
    except Exception as e:
        # Likely a timeout during fetching but had an error in requests.get()
        status = 500
        error_msg = '[Exception during download during fetching] ' + str(e)
        return dest_fname, status, error_msg

    if response.ok:
        try:
            with PathManager.open(os.path.join(path, dest_fname), 'wb+') as out_file:
                # Some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
            status = 200
        except Exception as e:
            # Likely a timeout during download or decoding
            status = 500
            error_msg = '[Exception during decoding or writing] ' + str(e)
    else:
        # We get here if there is an HTML error page (i.e. a page saying "404
        # not found" or anything else)
        status = response.status_code
        error_msg = '[Response not OK] Response: %s' % response

    return dest_fname, status, error_msg



# import parlai.core.build_data as build_data
# import parlai.utils.logging as logging
# from parlai.utils.io import PathManager

import os
from PIL import Image
import torch
from zipfile import ZipFile

_greyscale = '  .,:;crsA23hHG#98&@'
_cache_size = 84000

# Mapping from image mode to (torch_instantiation_str, layer_cutoff_idx)
IMAGE_MODE_SWITCHER = {
    'resnet152': ['resnet152', -1],
    'resnet101': ['resnet101', -1],
    'resnet50': ['resnet50', -1],
    'resnet34': ['resnet34', -1],
    'resnet18': ['resnet18', -1],
    'resnet152_spatial': ['resnet152', -2],
    'resnet101_spatial': ['resnet101', -2],
    'resnet50_spatial': ['resnet50', -2],
    'resnet34_spatial': ['resnet34', -2],
    'resnet18_spatial': ['resnet18', -2],
    'resnext101_32x8d_wsl': ['resnext101_32x8d_wsl', -1],
    'resnext101_32x16d_wsl': ['resnext101_32x16d_wsl', -1],
    'resnext101_32x32d_wsl': ['resnext101_32x32d_wsl', -1],
    'resnext101_32x48d_wsl': ['resnext101_32x48d_wsl', -1],
    'resnext101_32x8d_wsl_spatial': ['resnext101_32x8d_wsl', -2],
    'resnext101_32x16d_wsl_spatial': ['resnext101_32x16d_wsl', -2],
    'resnext101_32x32d_wsl_spatial': ['resnext101_32x32d_wsl', -2],
    'resnext101_32x48d_wsl_spatial': ['resnext101_32x48d_wsl', -2],
}


class ImageLoader:
    """
    Extract image feature using pretrained CNN network.
    """

    def __init__(self, opt):
        self.opt = opt.copy()
        self.use_cuda = False
        self.netCNN = None
        self.image_mode = opt.get('image_mode', 'no_image_model')
        self.use_cuda = not self.opt.get('no_cuda', False) and torch.cuda.is_available()
        if self.image_mode not in ['no_image_model', 'raw', 'ascii']:
            if 'image_mode' not in opt or 'image_size' not in opt:
                raise RuntimeError(
                    'Need to add image arguments to opt. See '
                    'parlai.core.params.ParlaiParser.add_image_args'
                )
            self.image_size = opt['image_size']
            self.crop_size = opt['image_cropsize']
            self._init_transform()
            if 'resnet' in self.image_mode:
                self._init_resnet_cnn()
            elif 'resnext' in self.image_mode:
                self._init_resnext_cnn()
            else:
                raise RuntimeError(
                    'Image mode {} not supported'.format(self.image_mode)
                )

    @classmethod
    def is_spatial(cls, image_mode: str):
        """
        Return if image mode has spatial dimensionality.
        """
        return any([s in image_mode for s in ['spatial']])

    def _init_transform(self):
        # initialize the transform function using torch vision.
        try:
            import torchvision
            import torchvision.transforms

            self.torchvision = torchvision
            self.transforms = torchvision.transforms

        except ImportError:
            raise ImportError('Please install torchvision; see https://pytorch.org/')

        self.transform = self.transforms.Compose(
            [
                self.transforms.Resize(self.image_size),
                self.transforms.CenterCrop(self.crop_size),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_resnet_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnet`` varieties
        """
        cnn_type, layer_num = self._image_mode_switcher()
        # initialize the pretrained CNN using pytorch.
        CNN = getattr(self.torchvision.models, cnn_type)

        # cut off the additional layer.
        self.netCNN = torch.nn.Sequential(
            *list(CNN(pretrained=True).children())[:layer_num]
        )

        if self.use_cuda:
            self.netCNN.cuda()

    def _init_resnext_cnn(self):
        """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnext101_..._wsl`` varieties
        """
        try:
            cnn_type, layer_num = self._image_mode_switcher()
            model = torch.hub.load('facebookresearch/WSL-Images', cnn_type)
            # cut off layer for ImageNet classification
            # if spatial, cut off another layer for spatial features
            self.netCNN = torch.nn.Sequential(*list(model.children())[:layer_num])
        except RuntimeError as e:
            # Perhaps specified one of the wrong model names
            model_names = [m for m in IMAGE_MODE_SWITCHER if 'resnext101' in m]
            logging.error(
                'If you have specified one of the resnext101 wsl models, '
                'please make sure it is one of the following: \n'
                f"{', '.join(model_names)}"
            )
            raise e
        except AttributeError:
            # E.g. "module 'torch' has no attribute 'hub'"
            raise RuntimeError(
                'Please install the latest pytorch distribution to have access '
                'to the resnext101 wsl models (pytorch 1.1.0, torchvision 0.3.0)'
            )

        if self.use_cuda:
            self.netCNN.cuda()

    def _image_mode_switcher(self):
        if self.image_mode not in IMAGE_MODE_SWITCHER:
            raise NotImplementedError(
                'image preprocessing mode'
                + '{} not supported yet'.format(self.image_mode)
            )

        return IMAGE_MODE_SWITCHER.get(self.image_mode)

    @classmethod
    def get_available_model_names(cls):
        """
        Get a list of the available model variants in this ImageLoader.
        """
        return list(IMAGE_MODE_SWITCHER.keys())

    def extract(self, image, path=None):
        # check whether initialize CNN network.
        # extract the image feature
        if 'faster_r_cnn' not in self.image_mode:
            transform = self.transform(image).unsqueeze(0)
            if self.use_cuda:
                transform = transform.cuda()
            with torch.no_grad():
                feature = self.netCNN(transform)
        else:
            raise RuntimeError("detectron support has been removed.")
        # save the feature
        if path is not None:
            # import parlai.utils.torch as torch_utils

            atomic_save(feature.cpu(), path)
        return feature

    def _img_to_ascii(self, im):
        im.thumbnail((60, 40), Image.BICUBIC)
        im = im.convert('L')
        asc = []
        for y in range(0, im.size[1]):
            for x in range(0, im.size[0]):
                lum = 255 - im.getpixel((x, y))
                asc.append(_greyscale[lum * len(_greyscale) // 256])
            asc.append('\n')
        return ''.join(asc)

    def _breakup_zip_filename(self, path):
        # assume format path/to/file.zip/image_name.jpg
        assert '.zip' in path
        sep = path.index('.zip') + 4
        zipname = path[:sep]
        file_name = path[sep + 1 :]
        return zipname, file_name

    def _get_prepath(self, path):
        if '.zip' in path:
            zipname, file_name = self._breakup_zip_filename(path)
            task = self.opt['task']
            prepath = os.path.join(self.opt['datapath'], task)
            imagefn = ''.join(zipname.strip('.zip').split('/')[-2:]) + path.name
            return prepath, imagefn
        else:
            prepath, imagefn = os.path.split(path)
            return prepath, imagefn

    def _load_image(self, path):
        """
        Return the loaded image in the path.
        """
        if '.zip' in path:
            zipname, file_name = self._breakup_zip_filename(path)
            with ZipFile(PathManager.open(zipname, 'rb')) as zipf:
                with zipf.open(file_name) as fh:
                    return Image.open(fh).convert('RGB')
        else:
            # raw just returns RGB values
            with PathManager.open(path, 'rb') as f:
                return Image.open(f).convert('RGB')

    def load(self, path):
        """
        Load from a given path.
        """
        mode = self.opt.get('image_mode', 'raw')
        if mode is None or mode == 'no_image_model':
            # don't need to load images
            return None
        elif mode == 'raw':
            return self._load_image(path)
        elif mode == 'ascii':
            # convert images to ascii ¯\_(ツ)_/¯
            return self._img_to_ascii(self._load_image(path))

        # otherwise, looks for preprocessed version under 'mode' directory
        prepath, imagefn = self._get_prepath(path)
        dpath = os.path.join(prepath, mode)
        if not PathManager.exists(dpath):
            make_dir(dpath)
        imagefn = imagefn.split('.')[0]
        new_path = os.path.join(prepath, mode, imagefn)
        if not PathManager.exists(new_path):
            return self.extract(self._load_image(path), new_path)
        else:
            with PathManager.open(new_path, 'rb') as f:
                return torch.load(f)


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Common Abstract classes for many agents.

This module provides a set of basic agents:

    ``Agent(object)``
    base class for all other agents, implements the ``observe()`` method
    which receives an observation/action dict and the ``act()`` method which
    returns a dict in response.

    ``Teacher(Agent)``
    also implements the ``report()`` method for returning metrics. All ParlAI
    tasks implement the ``Teacher`` class.

    ``MultiTaskTeacher(Teacher)``
    creates a set of teachers based on a task string passed to the ``Teacher``,
    creating multiple teachers within it and alternating between them.

All agents are initialized with the following parameters:

    ``opt`` -- contains any options needed to set up the agent. This generally contains
    all command-line arguments recognized from ``core.params``, as well as other
    options that might be set through the framework to enable certain modes.

    ``shared`` (optional) -- if not ``None``, contains any shared data used to construct
    this particular instantiation of the agent. This data might have been
    initialized by another agent, so that different agents can share the same
    data (possibly in different Processes).

This module also provides a utility method:

    ``create_task_agents(str)``: instantiate task-specific agents (e.g. a teacher)
    from a given task string (e.g. 'babi:task1k:1' or 'squad'). Used by
    ``MultiTaskTeacher``.
"""

import copy
from typing import List, Union

# from parlai.core.build_data import modelzoo_path
# from parlai.core.loader import load_agent_module
# from parlai.core.loader import register_agent  # noqa: F401
# from parlai.core.message import Message

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for Message object and associated functions.

The Message object's key function is to prevent users from editing fields in an action
or observation dict unintentionally.
"""
from typing import Any, Dict


UNSAFE_FIELDS = {'metrics'}


class Message(dict):
    """
    Class for observations and actions in ParlAI.

    Functions like a dict, but triggers a RuntimeError when calling __setitem__ for a
    key that already exists in the dict.
    """

    def __setitem__(self, key, val):
        if key in self:
            raise RuntimeError(
                'Message already contains key `{}`. If this was intentional, '
                'please use the function `force_set(key, value)`.'.format(key)
            )
        super().__setitem__(key, val)

    def force_set(self, key, val):
        super().__setitem__(key, val)

    def copy(self):
        return type(self)(self)

    @classmethod
    def padding_example(cls) -> Message:
        """
        Create a Message for batch padding.
        """
        return cls({'batch_padding': True, 'episode_done': True})

    def is_padding(self) -> bool:
        """
        Determine if a message is a padding example or not.
        """
        return bool(self.get('batch_padding'))

    def json_safe_payload(self) -> Dict[str, Any]:
        """
        Prepare a Message for delivery to a client via json.

        Useful for chat-services, external libraries, and mephisto delivery.

        Works by stripping known unsafe fields from the message, and converting
        the object to a dict.
        """
        return {k: v for k, v in self.items() if k not in UNSAFE_FIELDS}


# from parlai.core.opt import Opt
# from parlai.utils.misc import warn_once
# import parlai.utils.logging as logging
# from parlai.utils.io import PathManager


NOCOPY_ARGS = [
    'datapath',  # never use the datapath from an opt dump
    'batchindex',  # this saved variable can cause trouble if we switch to BS=1 at test time
]


class Agent(object):
    """
    Base class for all other agents.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Return an observation/action dict based upon given observation.
        """
        if hasattr(self, 'observation') and self.observation is not None:
            logging.info(f'agent received observation:\n{self.observation}')

        t = {}
        t['text'] = 'hello, teacher!'
        logging.info(f'agent sending message:\n{t}')
        return t

    def getID(self):
        """
        Return the agent ID.
        """
        return self.id

    def epoch_done(self):
        """
        Return whether the epoch is done or not.

        :rtype: boolean
        """
        return False

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None

    def reset_metrics(self):
        """
        Reset any metrics reported by this agent.

        This is called to indicate metrics should start fresh, and is typically called
        between loggings or after a `report()`.
        """
        pass

    def save(self, path=None):
        """
        Save any parameters needed to recreate this agent from loaded parameters.

        Default implementation is no-op, but many subagents implement this logic.
        """
        pass

    def clone(self):
        """
        Make a shared copy of this agent.

        Should be the same as using create_agent_from_shared(.), but slightly easier.
        """
        return type(self)(self.opt, self.share())

    def share(self):
        """
        Share any parameters needed to create a shared version of this agent.

        Default implementation shares the class and the opt, but most agents will want
        to also add model weights, teacher data, etc. This especially useful for
        avoiding providing pointers to large objects to all agents in a batch.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """
        Perform any final cleanup if needed.
        """
        pass

    def respond(
        self, text_or_message: Union[str, Message], **other_message_fields
    ) -> str:
        """
        An agent convenience function which calls the act() and provides a string
        response to a text or message field.

        :param Union[str, Message] text_or_message:
            A string for the 'text' field or a message which MUST
            comprise of the 'text' field apart from other fields.
        :param kwargs other_message_fields:
            Provide fields for the message in the form of keyword arguments.
        :return:
            Agent's response to the message.
        :rtype:
            str
        """
        if isinstance(text_or_message, str):
            observation = Message(text=text_or_message, **other_message_fields)
        else:
            observation = Message(**text_or_message, **other_message_fields)
            if 'text' not in observation:
                raise RuntimeError('The agent needs a \'text\' field in the message.')

        if 'episode_done' not in observation:
            observation['episode_done'] = True
        agent = self.clone()
        agent.observe(observation)
        response = agent.act()
        return response['text']

    def batch_respond(self, messages: List[Message]) -> List[str]:
        """
        An agent convenience function which calls the batch_act() and provides a batch
        response to a list of messages.

        :param List[Message] messages:
            A list of messages each of which MUST comprise of the 'text' field
            apart from other fields.
        :return:
            Agent's batch response to the messages.
        :rtype:
            List[str]
        """
        observations = []
        agents = []
        for i, message in enumerate(messages):
            if 'text' not in message:
                raise RuntimeError(
                    'The agent needs a \'text\' field in the {}th message.'.format(i)
                )
            if 'episode_done' not in message:
                message['episode_done'] = True
            agent = self.clone()
            agents.append(agent)
            observations.append(agent.observe(message))
        agent_acts = self.batch_act(observations)
        response = []
        for agent, resp in zip(agents, agent_acts):
            if hasattr(agent, "self_observe"):
                agent.self_observe(resp)
            response.append(resp['text'])
        return response

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        """
        Upgrade legacy options when loading an opt file from disk.

        This is primarily made available to provide a safe space to handle
        backwards-compatible behavior. For example, perhaps we introduce a
        new option today, which wasn't previously available. We can have the
        argument have a new default, but fall back to the "legacy" compatibility
        behavior if the option doesn't exist.

        ``upgrade_opt`` provides an opportunity for such checks for backwards
        compatibility. It is called shortly after loading the opt file from
        disk, and is called before the Agent is initialized.

        Other possible examples include:

            1. Renaming an option,
            2. Deprecating an old option,
            3. Splitting coupled behavior, etc.

        Implementations of ``upgrade_opt`` should conform to high standards,
        due to the risk of these methods becoming complicated and difficult to
        reason about. We recommend the following behaviors:

            1. ``upgrade_opt`` should only be used to provide backwards
            compatibility.  Other behavior should find a different location.
            2. Children should always call the parent's ``upgrade_opt`` first.
            3. ``upgrade_opt`` should always warn when an option was overwritten.
            4. Include comments annotating the date and purpose of each upgrade.
            5. Add an integration test which ensures your old work behaves
            appropriately.

        :param Opt opt_from_disk:
            The opt file, as loaded from the ``.opt`` file on disk.
        :return:
            The modified options
        :rtype:
            Opt
        """
        # 2019-07-11: currently a no-op.
        return opt_from_disk


def compare_init_model_opts(opt: Opt, curr_opt: Opt):
    """
    Print loud warning when `init_model` opts differ from previous configuration.
    """
    if opt.get('init_model') is None:
        return
    opt['init_model'] = modelzoo_path(opt['datapath'], opt['init_model'])
    optfile = opt['init_model'] + '.opt'
    if not PathManager.exists(optfile):
        return
    init_model_opt = Opt.load(optfile)

    extra_opts = {}
    different_opts = {}
    exempt_opts = [
        'model_file',
        'dict_file',
        'override',
        'starttime',
        'init_model',
        'batchindex',
    ]

    # search through init model opts
    for k, v in init_model_opt.items():
        if (
            k not in exempt_opts
            and k in init_model_opt
            and init_model_opt[k] != curr_opt.get(k)
        ):
            if isinstance(v, list):
                if init_model_opt[k] != list(curr_opt.get(k, [])):
                    different_opts[k] = ','.join([str(x) for x in v])
            else:
                different_opts[k] = v

    # search through opts to load
    for k, v in curr_opt.items():
        if k not in exempt_opts and k not in init_model_opt:
            if isinstance(v, list):
                extra_opts[k] = ','.join([str(x) for x in v])
            else:
                extra_opts[k] = v

    # print warnings
    extra_strs = ['{}: {}'.format(k, v) for k, v in extra_opts.items()]
    if extra_strs:
        logging.warning(
            'your model is being loaded with opts that do not '
            'exist in the model you are initializing the weights with: '
            '{}'.format(','.join(extra_strs))
        )

    different_strs = [
        '--{} {}'.format(k.replace('_', '-'), v) for k, v in different_opts.items()
    ]
    if different_strs:
        logging.warning(
            'your model is being loaded with opts that differ '
            'from the model you are initializing the weights with. Add the '
            'following args to your run command to change this: \n'
            '{}'.format(' '.join(different_strs))
        )


def create_agent_from_model_file(model_file, opt_overrides=None):
    """
    Load agent from model file if it exists.

    :param opt_overrides:
        An optional dict of option overrides can also be provided.
    :return:
        The agent
    """
    opt = {}
    add_datapath_and_model_args(opt)
    opt['model_file'] = modelzoo_path(opt.get('datapath'), model_file)
    if opt_overrides is None:
        opt_overrides = {}
    opt['override'] = opt_overrides
    return create_agent_from_opt_file(opt)


def create_agent_from_opt_file(opt: Opt):
    """
    Load agent options and module from file if opt file exists.

    Checks to see if file exists opt['model_file'] + ".opt"; if so, load up the
    options from the file and use that to create an agent, loading the model
    type from that file and overriding any options specified in that file when
    instantiating the agent.

    If that file does not exist, return None.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'

    if not PathManager.exists(optfile):
        return None

    opt_from_file = Opt.load(optfile)

    # delete args that we do not want to copy over when loading the model
    for arg in NOCOPY_ARGS:
        if arg in opt_from_file:
            del opt_from_file[arg]

    # only override opts specified in 'override' dict
    if opt.get('override'):
        for k, v in opt['override'].items():
            if k in opt_from_file and str(v) != str(opt_from_file.get(k)):
                logging.warning(
                    f'Overriding opt["{k}"] to {v} (previously: {opt_from_file.get(k)})'
                )
            opt_from_file[k] = v

    model_class = load_agent_module(opt_from_file['model'])

    if hasattr(model_class, 'upgrade_opt'):
        opt_from_file = model_class.upgrade_opt(opt_from_file)

    # add model arguments to opt_from_file if they aren't in opt_from_file already
    for k, v in opt.items():
        if k not in opt_from_file:
            opt_from_file[k] = v

    # update model file path to the one set by opt
    opt_from_file['model_file'] = model_file
    # update init model path to the one set by opt
    # NOTE: this step is necessary when for example the 'init_model' is
    # set by the Train Loop (as is the case when loading from checkpoint)
    if opt.get('init_model') is not None:
        opt_from_file['init_model'] = opt['init_model']

    # update dict file path
    if not opt_from_file.get('dict_file'):
        old_dict_file = None
        opt_from_file['dict_file'] = model_file + '.dict'
    elif opt_from_file.get('dict_file') and not PathManager.exists(
        opt_from_file['dict_file']
    ):
        old_dict_file = opt_from_file['dict_file']
        opt_from_file['dict_file'] = model_file + '.dict'
    if not PathManager.exists(opt_from_file['dict_file']):
        warn_once(
            'WARNING: Neither the specified dict file ({}) nor the '
            '`model_file`.dict file ({}) exists, check to make sure either '
            'is correct. This may manifest as a shape mismatch later '
            'on.'.format(old_dict_file, opt_from_file['dict_file'])
        )

    # if we want to load weights from --init-model, compare opts with
    # loaded ones
    compare_init_model_opts(opt, opt_from_file)
    return model_class(opt_from_file)


def add_datapath_and_model_args(opt: Opt):
    # add datapath, it is missing
    # from parlai.core.params import ParlaiParser, get_model_name

    parser = ParlaiParser(add_parlai_args=False)
    parser.add_parlai_data_path()
    # add model args if they are missing
    model = get_model_name(opt)
    if model is not None:
        parser.add_model_subargs(model, opt)
    opt_parser = parser.parse_args("")
    for k, v in opt_parser.items():
        if k not in opt:
            opt[k] = v


def create_agent(opt: Opt, requireModelExists=False):
    """
    Create an agent from the options ``model``, ``model_params`` and ``model_file``.

    The input is either of the form
    ``parlai.agents.ir_baseline.agents:IrBaselineAgent`` (i.e. the path
    followed by the class name) or else just ``ir_baseline`` which
    assumes the path above, and a class name suffixed with 'Agent'.

    If ``model-file`` is available in the options this function can also
    attempt to load the model from that location instead. This avoids having to
    specify all the other options necessary to set up the model including its
    name as they are all loaded from the options file if it exists (the file
    opt['model_file'] + '.opt' must exist and contain a pickled or json dict
    containing the model's options).
    """
    if opt.get('datapath', None) is None:
        add_datapath_and_model_args(opt)

    if opt.get('model_file'):
        opt['model_file'] = modelzoo_path(opt.get('datapath'), opt['model_file'])
        if requireModelExists and not PathManager.exists(opt['model_file']):
            raise RuntimeError(
                'WARNING: Model file does not exist, check to make '
                'sure it is correct: {}'.format(opt['model_file'])
            )
        # Attempt to load the model from the model file first (this way we do
        # not even have to specify the model name as a parameter)
        model = create_agent_from_opt_file(opt)
        if model is not None:
            return model
        else:
            logging.info(f"No model with opt yet at: {opt['model_file']}(.opt)")

    if opt.get('model'):
        model_class = load_agent_module(opt['model'])
        # if we want to load weights from --init-model, compare opts with
        # loaded ones
        compare_init_model_opts(opt, opt)
        model = model_class(opt)
        if requireModelExists and hasattr(model, 'load') and not opt.get('model_file'):
            # double check that we didn't forget to set model_file on loadable model
            logging.warning('model_file unset but model has a `load` function.')
        return model
    else:
        raise RuntimeError('Need to set `model` argument to use create_agent.')


# Helper functions to create agent/agents given shared parameters
# returned from agent.share(). Useful for parallelism, sharing params, etc.
def create_agent_from_shared(shared_agent):
    """
    Instantiate an agent from the default `shared` params.

    :param shared_agent:
        should include an `opt` dictionary and agent `class`, along with
        whatever other parameters the agent needs to instantiate.
    """
    opt = copy.deepcopy(shared_agent['opt'])
    a = shared_agent['class'](opt, shared_agent)
    return a


def create_agents_from_shared(shared):
    """
    Create agents based on shared data.

    :param shared: `list` of `dict` objects created by calling e.g.
        [a.share() for a in agents].

    Returns a list of instantiated agents.
    """
    shared_agents = []
    for shared_agent in shared:
        agent = create_agent_from_shared(shared_agent)
        shared_agents.append(agent)
    return shared_agents


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Worlds are the basic environments which define how agents interact with one another.

    ``World(object)`` provides a generic parent class, including ``__enter__``
    and ``__exit__`` statements which allow you to guarantee that the shutdown
    method is called.

    ``DialogPartnerWorld(World)`` provides a two-agent turn-based dialog setting.

    ``MultiAgentDialogWorld(World)`` provides a multi-agent setting.

    ``MultiWorld(World)`` creates a set of environments (worlds) for the same agent
    to multitask over, a different environment will be chosen per episode.

    ``BatchWorld(World)`` is a container for doing minibatch training over a world by
    collecting batches of N copies of the environment (each with different state).


All worlds are initialized with the following parameters:

    ``opt`` -- contains any options needed to set up the agent. This generally contains
        all command-line arguments recognized from core.params, as well as other
        options that might be set through the framework to enable certain modes.
    ``agents`` -- the set of agents that should be attached to the world,
        e.g. for DialogPartnerWorld this could be the teacher (that defines the
        task/dataset) and the learner agent. This is ignored in the case of
        sharing, and the shared parameter is used instead to initialize agents.
    ``shared`` (optional) -- if not None, contains any shared data used to construct
        this particular instantiation of the world. This data might have been
        initialized by another world, so that different agents can share the same
        data (possibly in different Processes).
"""

import copy
import random

from typing import Dict, List, Optional, Union

# import parlai.utils.logging as logging
# from parlai.core.agents import create_agents_from_shared
# from parlai.core.loader import load_task_module, load_world_module
# from parlai.core.metrics import (
#     aggregate_named_reports,
#     aggregate_unnamed_reports,
#     TeacherMetrics,
# )
# from parlai.core.opt import Opt
# from parlai.core.params import ParlaiParser
# from parlai.core.teachers import Teacher
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This module provides a set of teachers that deal with dialog.

    ``FixedDialogTeacher(Teacher)``
    Base class for teachers in tasks that have fixed dialog - i.e., dialog
    that is not dynamically generated but rather is pulled from set examples.
    However, the class can be extended to all tasks involved fixed data.
    Implements much of the basic functionality of these teachers, including
    ``observe()``, ``act()``, ``next_example()``

    ``DialogTeacher(FixedDialogTeacher)``
     Base teacher class for doing dialog specifically with fixed chat logs.

    ``ParlAIDialogTeacher(DialogTeacher)``
     Teacher class that provides access to data in the ParlAI Dialog format.
     See the class description for more details.

     ``ConversationTeacher(DialogTeacher)``
     Teacher class that provides access to data in the Conversations format.
     See the class description for more details.

    ``FbDeprecatedDialogTeacher(DialogTeacher)``
     Teacher class that provides access to data in the Facebook Dialog format.
     See the class description for more details. **This class is deprecated**.

This module also includes ``DataLoader``, a threadpool data loader for
``FixedDialogTeacher``, and ``DialogData``/``StreamDialogData``, data
structures for accessing textual dialog data and utilized by ``DialogTeacher``
"""
# from parlai.core.params import ParlaiParser
# from parlai.core.agents import Agent, create_agent_from_shared
# from parlai.core.image_featurizers import ImageLoader
# from parlai.core.loader import load_teacher_module
# from parlai.core.loader import register_teacher  # noqa: F401
# from parlai.core.message import Message
# from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
# from parlai.core.opt import Opt
# from parlai.utils.conversations import Conversations

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility functions and classes for handling text strings.
"""
import os
import sys as _sys


def normalize_reply(text: str, version=1) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.

    Version 1: Fix sentence start casing, and punctuation.

    Version 2: Add trailing period, if missing.
    """

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]


def name_to_classname(name: str) -> str:
    words = name.split('_')
    class_name = ''
    for w in words:
        # capitalize the first letter
        class_name += w[0].upper() + w[1:]
    return class_name


def colorize(text, style):
    try:
        # if we're in ipython it's okay to use colors
        __IPYTHON__
        USE_COLORS = True
    except NameError:
        USE_COLORS = _sys.stdout.isatty()

    if not USE_COLORS:
        return text

    colorstyle = os.environ.get('PARLAI_COLORSTYLE')

    RESET = '\033[0;0m'
    if style == 'red':
        return '\033[0;31m' + text + RESET
    if style == 'yellow':
        return '\033[0;93m' + text + RESET
    if style == 'green':
        return '\033[0;32m' + text + RESET
    if style == 'blue':
        return '\033[0;34m' + text + RESET
    if style == 'brightblack':
        return '\033[0;90m' + text + RESET

    if colorstyle is None or colorstyle.lower() == 'steamroller':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY_NOBK = '\033[1m'
        LIGHT_GRAY_NOBK = '\033[0m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED_NOBK = '\033[1;31m'
        HIGHLIGHT_BLUE_NOBK = '\033[0;34m'
        if style == 'highlight':
            return HIGHLIGHT_RED_NOBK + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        elif style == 'text':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY_NOBK + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'id':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        else:
            return MAGENTA + text + RESET

    if colorstyle.lower() == 'spermwhale':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY = '\033[1;37;40m'
        LIGHT_GRAY = '\033[0;37;40m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED = '\033[1;37;41m'
        HIGHLIGHT_BLUE = '\033[1;37;44m'
        if style == 'highlight':
            return HIGHLIGHT_RED + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE + text + RESET
        elif style == 'text':
            return LIGHT_GRAY + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY + text + RESET
        elif style == 'id':
            return LIGHT_GRAY + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE + text + RESET
        else:
            return MAGENTA + text + RESET

    # No colorstyle specified/found.
    return text


"""
Utility methods for conversations format.
"""
import datetime
import json
import os
import itertools

# from parlai.utils.io import PathManager
# from parlai.core.metrics import dict_report
# from parlai.utils.misc import AttrDict
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
File for miscellaneous utility functions and constants.
"""

from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
import os

# from parlai.core.message import Message
# from parlai.utils.strings import colorize

# from parlai.utils.io import PathManager
# import parlai.utils.logging as logging

try:
    import torch

    __TORCH_AVAILABLE = True
except ImportError:
    # silence the error, we'll have other problems later if it's super necessary
    __TORCH_AVAILABLE = False


SPECIAL_FORMATED_DISPLAY_MESSAGE_FIELDS = {
    'episode_done',
    'id',
    'image',
    'text',
    'labels',
    'eval_labels',
    'label_candidates',
    'text_candidates',
    'reward',
    'token_losses',
    'metrics',
}

MUST_SHOW_MESSAGE_FIELDS = {'image', 'text', 'labels', 'eval_labels', 'reward'}


def maintain_dialog_history(
    history,
    observation,
    reply='',
    historyLength=1,
    useReplies='label_else_model',
    dict=None,
    useStartEndIndices=True,
    splitSentences=False,
):
    """
    Keep track of dialog history, up to a truncation length.

    Either includes replies from the labels, model, or not all using param
    'replies'.

    DEPRECATED. USE PARLAI.CORE.TORCH_AGENT INSTEAD.
    """

    def parse(txt, splitSentences):
        if dict is not None:
            if splitSentences:
                vec = [dict.txt2vec(t) for t in txt.split('\n')]
            else:
                vec = dict.txt2vec(txt)
            return vec
        else:
            return [txt]

    if 'dialog' not in history:
        history['dialog'] = deque(maxlen=historyLength)
        history['episode_done'] = False
        history['labels'] = []

    if history['episode_done']:
        history['dialog'].clear()
        history['labels'] = []
        useReplies = 'none'
        history['episode_done'] = False

    if useReplies != 'none':
        if useReplies == 'model' or (
            useReplies == 'label_else_model' and len(history['labels']) == 0
        ):
            if reply:
                if useStartEndIndices:
                    reply = dict.start_token + ' ' + reply
                history['dialog'].extend(parse(reply, splitSentences))
        elif len(history['labels']) > 0:
            r = history['labels'][0]
            history['dialog'].extend(parse(r, splitSentences))

    obs = observation
    if 'text' in obs:
        if useStartEndIndices:
            obs['text'] = dict.end_token + ' ' + obs['text']
        history['dialog'].extend(parse(obs['text'], splitSentences))

    history['episode_done'] = obs['episode_done']

    labels = obs.get('labels', obs.get('eval_labels', None))
    if labels is not None:
        if useStartEndIndices:
            history['labels'] = [dict.start_token + ' ' + l for l in labels]
        else:
            history['labels'] = labels

    return history['dialog']


def load_cands(path, lines_have_ids=False, cands_are_replies=False):
    """
    Load global fixed set of candidate labels that the teacher provides.

    Every example will include these as candidates. The true labels for a specific
    example are also added to this set, so that it's possible to get the right answer.
    """
    if path is None:
        return None
    cands = []
    cnt = 0
    with PathManager.open(path) as read:
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) > 0:
                cnt = cnt + 1
                # If lines are numbered we strip them of numbers.
                if cnt == 1 and line[0:2] == '1 ':
                    lines_have_ids = True
                # If tabs then the label_candidates are all the replies.
                if '\t' in line and not cands_are_replies:
                    cands_are_replies = True
                    cands = []
                if lines_have_ids:
                    space_idx = line.find(' ')
                    line = line[space_idx + 1 :]
                    if cands_are_replies:
                        sp = line.split('\t')
                        if len(sp) > 1 and sp[1] != '':
                            cands.append(sp[1])
                    else:
                        cands.append(line)
                else:
                    cands.append(line)
    return cands


class Timer(object):
    """
    Computes elapsed time.
    """

    def __init__(self):
        """
        Initialize timer.
        """
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        """
        Reset timer to zero.
        """
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        """
        Resume timer.
        """
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """
        Pause timer.
        """
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        """
        Get current timer time.
        """
        if self.running:
            return self.total + time.time() - self.start
        return self.total

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides standard metric evaluations for dialog, as well as an aggregator.
"""

# from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
import math
from typing import (
    Any,
    Counter as TCounter,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch

# from parlai.core.message import Message
# from parlai.utils.misc import warn_once
# from parlai.utils.typing import TScalar, TVector


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definitions of general ParlAI types.
"""
from typing import Any, Dict, TypeVar, Union, List

import torch


class _Shared(Dict[str, Any]):
    """
    ParlAI ``shared`` Structure.

    The `shared` dict that is used to instantiate shared agents in ParlAI,
    e.g. when using batching, distributed training, etc.

    Type is ``TShared``.
    """


TShared = TypeVar('TShared', bound=_Shared)

TScalar = Union[int, float, torch.Tensor]
"""
ParlAI type to represent an object that is theoretically expressible as a scalar value.
Ints and floats are clearly scalars, and torch.Tensors can be represented by a scalar if
Tensor.numel() == 1. Used as input type for classes derived from Metric.
"""

TVector = Union[List[TScalar], torch.Tensor]



DEFAULT_METRICS = {'bleu-4', 'accuracy', 'f1'}
ROUGE_METRICS = {'rouge-1', 'rouge-2', 'rouge-L'}
ROUGE_METRICS_MEASURES = {'r', 'f', 'p'}
BLEU_METRICS = {'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'}
DISTINCT_METRICS = {
    'interdistinct-1',
    'interdistinct-2',
    'intradistinct-1',
    'intradistinct-2',
}
ALL_METRICS = DEFAULT_METRICS | ROUGE_METRICS | BLEU_METRICS | DISTINCT_METRICS


class MetricDisplayData(NamedTuple):
    title: str
    description: str


METRICS_DISPLAY_DATA = {
    "accuracy": MetricDisplayData("Accuracy", "Exact match text accuracy"),
    'auc': MetricDisplayData(
        'AUC',
        "Area Under the Receiver Operating Characteristic Curve (true positive rate vs false positive rate curve)",
    ),
    "bleu-4": MetricDisplayData(
        "BLEU-4",
        "BLEU-4 of the generation, under a standardized (model-independent) tokenizer",
    ),
    "clen": MetricDisplayData(
        "Context Length", "Average length of context in number of tokens"
    ),
    "clip": MetricDisplayData(
        "Clipped Gradients", "Fraction of batches with clipped gradients"
    ),
    "ctpb": MetricDisplayData("Context Tokens Per Batch", "Context tokens per batch"),
    "ctps": MetricDisplayData("Context Tokens Per Second", "Context tokens per second"),
    "ctrunc": MetricDisplayData(
        "Context Truncation", "Fraction of samples with some context truncation"
    ),
    "ctrunclen": MetricDisplayData(
        "Context Truncation Length", "Average length of context tokens truncated"
    ),
    "exps": MetricDisplayData("Examples Per Second", "Examples per second"),
    "exs": MetricDisplayData(
        "Examples", "Number of examples processed since last print"
    ),
    "f1": MetricDisplayData(
        "F1", "Unigram F1 overlap, under a standardized (model-independent) tokenizer"
    ),
    "gen_n_toks": MetricDisplayData(
        "Generation Length", "Average length of generated outputs in number of tokens"
    ),
    "gnorm": MetricDisplayData("Gradient Norm", "Gradient norm"),
    "gpu_mem": MetricDisplayData(
        "GPU Memory",
        "Fraction of GPU memory used. May slightly underestimate true value.",
    ),
    "hits@1": MetricDisplayData(
        "Hits@1", "Fraction of correct choices in 1 guess. (Similar to recall@K)"
    ),
    "hits@5": MetricDisplayData(
        "Hits@5", "Fraction of correct choices in 5 guesses. (Similar to recall@K)"
    ),
    "interdistinct-1": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "interdistinct-2": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "intradistinct-1": MetricDisplayData(
        "Intradictinct-1", "Fraction of n-grams unique _within_ each utterance"
    ),
    "intradictinct-2": MetricDisplayData(
        "Intradictinct-2", "Fraction of n-grams unique _within_ each utterance"
    ),
    "jga": MetricDisplayData("Joint Goal Accuracy", "Joint Goal Accuracy"),
    "llen": MetricDisplayData(
        "Label Length", "Average length of label in number of tokens"
    ),
    "loss": MetricDisplayData("Loss", "Loss"),
    "lr": MetricDisplayData("Learning Rate", "The most recent learning rate applied"),
    "ltpb": MetricDisplayData("Label Tokens Per Batch", "Label tokens per batch"),
    "ltps": MetricDisplayData("Label Tokens Per Second", "Label tokens per second"),
    "ltrunc": MetricDisplayData(
        "Label Truncation", "Fraction of samples with some label truncation"
    ),
    "ltrunclen": MetricDisplayData(
        "Label Truncation Length", "Average length of label tokens truncated"
    ),
    "precision": MetricDisplayData(
        "Precision",
        "Precision computed based on unigram, under a standardized (model-independent) tokenizer",
    ),
    "recall": MetricDisplayData(
        "Recall",
        "Recall computed based on unigram, under a standardized (model-independent) tokenizer",
    ),
    "rouge-1": MetricDisplayData("ROUGE-1", "ROUGE metrics"),
    "rouge-2": MetricDisplayData("ROUGE-2", "ROUGE metrics"),
    "rouge-L": MetricDisplayData("ROUGE-L", "ROUGE metrics"),
    "token_acc": MetricDisplayData(
        "Token Accuracy", "Token-wise accuracy (generative only)"
    ),
    "token_em": MetricDisplayData(
        "Token Exact Match",
        "Utterance-level token accuracy. Roughly corresponds to perfection under greedy search (generative only)",
    ),
    "total_train_updates": MetricDisplayData(
        "Total Train Updates", "Number of SGD steps taken across all batches"
    ),
    "tpb": MetricDisplayData(
        "Tokens Per Batch", "Total tokens (context + label) per batch"
    ),
    "tps": MetricDisplayData(
        "Tokens Per Second", "Total tokens (context + label) per second"
    ),
    "ups": MetricDisplayData("Updates Per Second", "Updates per second (approximate)"),
}


def get_metric_display_data(metric: str) -> MetricDisplayData:
    return METRICS_DISPLAY_DATA.get(
        metric,
        MetricDisplayData(
            title=metric,
            description="No description provided. Please add it to metrics.py if this is an official metric in ParlAI.",
        ),
    )


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


@functools.total_ordering  # type: ignore
class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @property
    def is_global(self) -> bool:
        """
        Indicates whether this metric should be reported globally or per-task.
        """
        return False

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    @abstractmethod
    def __add__(self, other: Any) -> Metric:
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__sub__ is intentionally limited to floats.')
        return self.value() - other

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.

        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__rsub__ is intentionally limited to floats.')
        return other - self.value()

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float)
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))

    @classmethod
    def many(cls, *objs: List[TVector]) -> List[Metric]:
        """
        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denominators, etc.
        """
        lengths = [len(o) for o in objs]
        objs = list(objs)  # convert from tuple for inplace modification
        for i, o in enumerate(objs):
            if isinstance(o, torch.Tensor):
                # if the tensor is on GPU, make sure we transfer the whole thing
                # at once, instead of one-element-at-a-time during our list
                # comprehension
                objs[i] = o.tolist()
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]

    @classmethod
    def from_mask(
        cls, metric_per_token: torch.Tensor, mask: torch.Tensor
    ) -> List[Metric]:
        """
        From token-level metrics, returns an aggregate MyMetric per example in the
        batch.

        :param metric_per_token:
            a (batchsize x num_tokens) Tensor
        :param mask:
            a (batchsize x num_tokens) Tensor to mask out tokens that should *not* be considered in the aggregate metric calculation.
        :return:
            a (batchsize) Tensor
        """
        tokens_per_ex = mask.long().sum(dim=-1)
        metric_per_ex = (metric_per_token * mask).sum(dim=-1)
        metrics = cls.many(metric_per_ex, tokens_per_ex)
        return metrics


class FixedMetric(Metric):
    """
    Fixed metrics are verified to be the same when combined, or throw an error.

    FixedMetric is used for things like total_train_updates, which should not be
    combined across different multitasks or different workers.
    """

    __slots__ = ('_value',)

    def __init__(self, value: TScalar):
        self._value = self.as_number(value)

    def __add__(self, other: Optional[FixedMetric]) -> FixedMetric:
        if other is None:
            return self
        if self != other:
            raise ValueError(f"FixedMetrics not the same: {self} and {other}")
        return self

    def value(self) -> float:
        return self._value


class SumMetric(Metric):
    """
    Class that keeps a running sum of some metric.

    Examples of SumMetric include things like "exs", the number of examples seen since
    the last report, which depends exactly on a teacher.
    """

    __slots__ = ('_sum',)

    def __init__(self, sum_: TScalar = 0):
        if isinstance(sum_, torch.Tensor):
            self._sum = sum_.item()
        else:
            assert isinstance(sum_, (int, float))
            self._sum = sum_

    def __add__(self, other: Optional[SumMetric]) -> SumMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_sum = self._sum + other._sum
        # always keep the same return type
        return type(self)(sum_=full_sum)

    def value(self) -> float:
        return self._sum


class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ('_numer', '_denom')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional[AverageMetric]) -> AverageMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom


class MacroAverageMetric(Metric):
    """
    Class that represents the macro average of several numbers.

    Used for aggregating task level metrics. It is only used for things that are
    AverageMetrics already.
    """

    __slots__ = '_values'

    def __init__(self, metrics: Dict[str, Metric]) -> None:
        self._values = metrics

    def __add__(self, other: Optional[MacroAverageMetric]) -> MacroAverageMetric:
        if other is None:
            return self
        output = dict(**self._values)
        for k, v in other._values.items():
            output[k] = output.get(k, None) + v
        return MacroAverageMetric(output)

    def value(self) -> float:
        sum_ = sum(v.value() for v in self._values.values())
        n = len(self._values)
        return sum_ / n


class TimerMetric(Metric):
    """
    A timer metric keep tracks of the first/last times it was used.
    """

    __slots__ = ('_value', '_start', '_end')

    @classmethod
    def _now(cls) -> float:
        return datetime.datetime.utcnow().timestamp()

    def __init__(
        self,
        value: TScalar,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self._value = self.as_number(value)
        if start_time is None:
            start_time = self._now()
        if end_time is None:
            end_time = self._now()
        self._start = start_time
        self._end = end_time

    def __add__(self, other: Optional[TimerMetric]) -> TimerMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        total: TScalar = self._value + other._value
        start: float = min(self._start, other._start)
        end: float = max(self._end, other._end)
        return type(self)(total, start, end)

    def value(self) -> float:
        if self._value == 0 or self._end == self._start:
            return 0
        return self._value / (self._end - self._start)


class GlobalMetric:
    """
    A global metric is one that should not be aggregated across different tasks.

    Examples of global metric include things like learning rate and updates.
    These need to be accumulated or averaged over multiple parleys, but cannot
    be correlated with a single task.

    Key to it is the notion that any one worker or any one task already has a global
    view of the value, and so no combinations should be done. Note this is different
    then a FixedMetric, in that a GlobalMetric can be still averaged across multiple
    parleys(), but a FixedMetric is always fixed.
    """

    @property
    def is_global(self) -> bool:
        return True


class GlobalFixedMetric(GlobalMetric, FixedMetric):
    """
    Global fixed metric.

    Used for things like total_train_updates.
    """

    pass


class GlobalSumMetric(GlobalMetric, SumMetric):
    """
    Global sum metric.

    Used for 'exs' and 'updates'.
    """

    pass


class GlobalAverageMetric(GlobalMetric, AverageMetric):
    """
    Global Average metric.

    Used for things like learning rate, and many agent-specific metrics.
    """

    pass


class LegacyMetric(GlobalAverageMetric):
    """
    Legacy Metrics are reported by agent as float.
    """

    pass


class GlobalTimerMetric(GlobalMetric, TimerMetric):
    pass


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(
        guess: str, answers: List[str], expose_p_and_r: bool = False
    ) -> Union[F1Metric, Tuple[F1Metric, F1Metric, F1Metric]]:
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        max_p, max_r, max_f1 = 0, 0, 0
        for p, r, f1 in scores:
            max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
        if expose_p_and_r:
            return (F1Metric(max_p, 1), F1Metric(max_r, 1), F1Metric(max_f1, 1))
        else:
            return F1Metric(max_f1, 1)


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> ExactMatchMetric:
        if guess is None or answers is None:
            return None
        guess = normalize_answer(guess)
        for a in answers:
            if guess == normalize_answer(a):
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int = 4) -> Optional[BleuMetric]:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        try:
            from nltk.translate import bleu_score as nltkbleu
        except ImportError:
            # User doesn't have nltk installed, so we can't use it for bleu
            # We'll just turn off things, but we might want to warn the user
            return None

        # Warning: BLEU calculation *should* include proper tokenization and
        # punctuation etc. We're using the normalize_answer for everything though,
        # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
        # going to be slower than fairseq's (which is written in C), but fairseq's
        # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
        # works with strings, which is better suited for this module.
        weights = [1 / k for _ in range(k)]
        score = nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return BleuMetric(score)


class FairseqBleuMetric(Metric):
    """
    Re-implementation of
    https://github.com/pytorch/fairseq/blob/main/fairseq/scoring/bleu.py.
    """

    def __init__(
        self,
        pred: Union[torch.Tensor, List[int]],
        ref: Union[torch.Tensor, List[int]],
        pad_idx: int,
        eos_idx: int,
        unk_idx: int,
        order: int,
    ):
        try:
            from fairseq import libbleu
            from fairseq.scoring.bleu import BleuStat
            import ctypes
        except ImportError:
            return

        self.stat = BleuStat()
        self.order = order

        C = ctypes.cdll.LoadLibrary(libbleu.__file__)
        C.bleu_zero_init(ctypes.byref(self.stat))

        if not torch.is_tensor(pred):
            pred = torch.LongTensor(pred)
        if not torch.is_tensor(ref):
            ref = torch.LongTensor(ref)

        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(unk_idx)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(pad_idx),
            ctypes.c_int(eos_idx),
        )

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __add__(self, other: Optional[FairseqBleuMetric]) -> FairseqBleuMetric:
        if other is None:
            return self
        self.stat.match1 += other.stat.match1
        self.stat.match2 += other.stat.match2
        self.stat.match3 += other.stat.match3
        self.stat.match4 += other.stat.match4
        self.stat.count1 += other.stat.count1
        self.stat.count2 += other.stat.count2
        self.stat.count3 += other.stat.count3
        self.stat.count4 += other.stat.count4
        self.stat.predlen += other.stat.predlen
        self.stat.reflen += other.stat.reflen
        return self

    def _ratio(self, a: int, b: int) -> float:
        """
        Safe division.
        """
        return a / b if b > 0 else 0

    def _precision(self):
        return [
            self._ratio(self.stat.match1, self.stat.count1),
            self._ratio(self.stat.match2, self.stat.count2),
            self._ratio(self.stat.match3, self.stat.count3),
            self._ratio(self.stat.match4, self.stat.count4),
        ]

    def _brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def value(self) -> float:
        """
        Reimplementation of Fairseq's score.
        """
        psum = sum(
            math.log(p) if p > 0 else float("-Inf")
            for p in self._precision()[: self.order]
        )
        return self._brevity() * math.exp(psum / self.order) * 100

    @staticmethod
    def compute_many(
        guess: torch.Tensor, answers: torch.Tensor, pad_idx, end_idx, unk_idx
    ):
        """
        Return BLEU-1..4 using fairseq and tokens.
        """
        try:
            from fairseq.scoring import bleu as fairseqbleu  # noqa
        except ImportError:
            return None

        return [
            FairseqBleuMetric(
                guess.cpu().int(),
                answers.cpu().int(),
                pad_idx,
                end_idx,
                unk_idx,
                order=i,
            )
            for i in range(1, 5)
        ]


class RougeMetric(AverageMetric):
    _evaluator = None

    @staticmethod
    def compute_many(
        guess: str, answers: List[str], measure: str = 'r'
    ) -> Tuple[Optional[RougeMetric], Optional[RougeMetric], Optional[RougeMetric]]:
        """
        Compute ROUGE score between guess and *any* answer.

        Done with compute_many due to increased efficiency.

        :return: (rouge-1, rouge-2, rouge-L)
        """
        measure = measure.lower()
        assert (
            measure in ROUGE_METRICS_MEASURES
        ), "Use one of recall 'r' (default), f1 'f', or precision 'p'."

        # possible global initialization
        try:
            import rouge
        except ImportError:
            # User doesn't have py-rouge installed, so we can't use it.
            # We'll just turn off rouge computations
            return None, None, None

        if RougeMetric._evaluator is None:
            RougeMetric._evaluator = rouge.Rouge(
                metrics=['rouge-n', 'rouge-l'], max_n=2
            )
        try:
            scores = [
                RougeMetric._evaluator.get_scores(
                    normalize_answer(guess), normalize_answer(a)
                )
                for a in answers
            ]
        except LookupError:
            warn_once(
                'ROUGE requires nltk punkt tokenizer. Please run '
                '`python -c "import nltk; nltk.download(\'punkt\')`'
            )
            return None, None, None

        scores_rouge1 = max(score['rouge-1'][measure] for score in scores)
        scores_rouge2 = max(score['rouge-2'][measure] for score in scores)
        scores_rougeL = max(score['rouge-l'][measure] for score in scores)
        return (
            RougeMetric(scores_rouge1),
            RougeMetric(scores_rouge2),
            RougeMetric(scores_rougeL),
        )


class IntraDistinctMetric(AverageMetric):
    """
    Compute intra-distinct (per-utterance).
    """

    @classmethod
    def _ngram(cls, seq, n: int):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text: str, ngram: int = 1):
        """
        :param text:
            The text to compute metric over
        :param ngram:
            n-gram length
        """
        tokens = normalize_answer(text).split()
        counts: Counter[Any] = Counter(cls._ngram(tokens, ngram))
        # computed per-example, macro averaged across examples
        intra = max(len(counts), 1e-12) / max(sum(counts.values()), 1e-5)
        return IntraDistinctMetric(intra, 1.0)


class InterDistinctMetric(Metric):
    """
    Compute inter-distinct metric over corpus-level.
    """

    def __init__(self, counts: TCounter[Tuple]):
        """
        :param counts:
            collections.Counter of ngram -> frequency
        """
        self._counts = counts

    def __add__(self, other):
        return InterDistinctMetric(self._counts + other._counts)

    def value(self):
        return max(len(self._counts), 1e-12) / max(sum(self._counts.values()), 1e-5)

    @classmethod
    def _ngram(cls, seq, n):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text, ngram=1):
        tokens = normalize_answer(text).split()
        return InterDistinctMetric(Counter(cls._ngram(tokens, ngram)))


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def aggregate_named_reports(
    named_reports: Dict[str, Dict[str, Metric]], micro_average: bool = False
) -> Dict[str, Metric]:
    """
    Aggregate metrics from multiple reports.

    :param reports:
        Dict of tasks -> metrics.
    :param micro_average:
        If true, top level metrics will be the micro average. By default, we
        use macro average.
    :return:
        The aggregated report
    """
    if len(named_reports) == 0:
        raise ValueError("Cannot aggregate empty reports.")
    if len(named_reports) == 1:
        # no real aggregation to be done
        return next(iter(named_reports.values()))

    # reporters is a list of teachers or worlds
    m: Dict[str, Metric] = {}
    macro_averages: Dict[str, Dict[str, Metric]] = {}
    for task_id, task_report in named_reports.items():
        for each_metric, value in task_report.items():
            if value.is_global:
                # just take the first one we saw
                if each_metric not in m:
                    m[each_metric] = value
            else:
                task_metric = f'{task_id}/{each_metric}'
                m[task_metric] = m.get(task_metric) + value
                if micro_average or not value.macro_average:
                    # none + a => a from implementation of Metric.__add__
                    m[each_metric] = m.get(each_metric) + value
                else:
                    # macro average
                    if each_metric not in macro_averages:
                        macro_averages[each_metric] = {}
                    macro_averages[each_metric][task_id] = value
    for key, values in macro_averages.items():
        m[key] = MacroAverageMetric(values)
    return m


def aggregate_unnamed_reports(reports: List[Dict[str, Metric]]) -> Dict[str, Metric]:
    """
    Combines metrics without regard for tracking provenence.
    """
    m: Dict[str, Metric] = {}
    for task_report in reports:
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric) + value
    return m


def dict_report(report: Dict[str, Metric]):
    return {k: v.value() if isinstance(v, Metric) else v for k, v in report.items()}


class Metrics(object):
    """
    Metrics aggregator.
    """

    def __init__(self, threadsafe=False, shared=None):
        if shared and 'data' in shared:
            # This is a clone
            self._data = shared['data']
        else:
            # The original
            self._data = {}

        # recent data is to track per-example metrics, and so should never be
        # shared
        self._recent_data = {}

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f'Metrics({repr(self._data)})'

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        self._data[key] = self._data.get(key) + value
        self._recent_data[key] = self._recent_data.get(key) + value

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        return self._data.copy()

    def clear_recent(self):
        """
        Clear recent metrics (latest example).
        """
        self._recent_data.clear()

    def report_recent(self):
        """
        Report recent metrics (latest example).
        """
        return self._recent_data.copy()

    def clear(self):
        """
        Clear all the metrics.
        """
        self._data.clear()
        self._recent_data.clear()

    def share(self):
        return {'data': self._data}

    def add_metrics(self, other: "Metrics") -> None:
        """
        Aggregate another Metrics objects metrics into this one.

        Note that it is assumed that the keys for metrics are disjoint between Metrics
        objects.
        """
        for k, v in other._data.items():
            self.add(k, v)


class TeacherMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(
        self, metrics_list: str = "default", shared: Dict[str, Any] = None
    ) -> None:
        super().__init__(shared=shared)
        self._metrics_list = self._infer_metrics(metrics_list)
        self.eval_pr = [1, 5, 10, 100]

    @staticmethod
    def _infer_metrics(cli_arg: str) -> Set[str]:
        """
        Parse the CLI metric into a list of metrics we wish to compute.
        """
        col: Set[str] = set()
        names = cli_arg.split(",")
        for n in names:
            if n == 'default':
                col |= DEFAULT_METRICS
            elif n == 'rouge':
                col |= ROUGE_METRICS
            elif n == 'bleu':
                col |= BLEU_METRICS
            elif n == 'distinct':
                col |= DISTINCT_METRICS
            elif n == 'all':
                col |= ALL_METRICS
            else:
                col.add(n)
        return col

    def _update_ranking_metrics(self, observation, labels):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            return

        # Now loop through text candidates, assuming they are sorted.
        # If any of them is a label then score a point.
        # maintain hits@1, 5, 10, 50, 100,  etc.
        label_set = set(normalize_answer(l) for l in labels)
        cnts = {k: 0 for k in self.eval_pr}
        cnt = 0
        for c in text_cands:
            cnt += 1
            if normalize_answer(c) in label_set:
                for k in self.eval_pr:
                    if cnt <= k:
                        cnts[k] += 1
        # hits metric is 1 if cnts[k] > 0.
        # (other metrics such as p@k and r@k take
        # the value of cnt into account.)
        for k in self.eval_pr:
            self.add(f'hits@{k}', AverageMetric(cnts[k] > 0))

    def evaluate_response(self, observation: Message, labels: List[str]) -> None:
        """
        Compute all required text-based metrics based on an observation and labels.
        """
        prediction = observation.get('text', None)

        self.add('exs', SumMetric(1))

        if prediction is not None:
            self.add('accuracy', ExactMatchMetric.compute(prediction, labels))
            precision, recall, f1 = F1Metric.compute(
                prediction, labels, expose_p_and_r=True
            )
            self.add('precision', precision)
            self.add('recall', recall)
            self.add('f1', f1)

            for k in range(1, 5):  # 1..4
                if f'bleu-{k}' in self._metrics_list:
                    self.add(f'bleu-{k}', BleuMetric.compute(prediction, labels, k))
            # if any of the rouges are in the list
            if self._metrics_list & ROUGE_METRICS:
                r1, r2, rL = RougeMetric.compute_many(prediction, labels)
                if 'rouge-1' in self._metrics_list and r1:
                    self.add('rouge_1', r1)
                if 'rouge-2' in self._metrics_list and r2:
                    self.add('rouge_2', r2)
                if 'rouge-L' in self._metrics_list and rL:
                    self.add('rouge_L', rL)
            # compute distinct-k
            for k in [1, 2]:
                if f'interdistinct-{k}' in self._metrics_list:
                    self.add(
                        f'interdistinct-{k}', InterDistinctMetric.compute(prediction, k)
                    )
                if f'intradistinct-{k}' in self._metrics_list:
                    self.add(
                        f'intradistinct-{k}', IntraDistinctMetric.compute(prediction, k)
                    )

        # Ranking metrics.
        self._update_ranking_metrics(observation, labels)

        self._consume_user_metrics(observation)

    def _consume_user_metrics(self, observation):
        # User-reported metrics
        if 'metrics' in observation:
            for uk, v in observation['metrics'].items():
                if v is None:
                    continue
                if uk in ALL_METRICS:
                    # don't let the user override our metrics
                    uk = f'USER_{uk}'
                assert isinstance(uk, str), f'{type(uk)} is not a str'
                if not isinstance(v, Metric):
                    warn_once(f'Metric {uk} is assumed to be averaged per example.')
                    v = AverageMetric(v)
                assert isinstance(v, Metric)
                self.add(uk, v)



class TimeLogger:
    """
    Class for logging time progress against a goal.
    """

    def __init__(self):
        """
        Set up timer.
        """
        self.timer = Timer()
        self.tot_time = 0

    def total_time(self):
        """
        Return time elapsed at last log call.
        """
        return self.tot_time

    def time(self):
        """
        Return current timer time.
        """
        return self.timer.time()

    def log(self, done, total, report=None):
        """
        Log report, time elapsed, and percentage progress towards goal.

        :param done: number of examples completed so far
        :param total: total number of elements to be completed. if total > 0,
                      calculates the time remaining and percentage complete.
        :param report: dict of pairs to log

        :returns: tuple log string, log dict
            log string contains time elapsed and string representation of
            the log dict
            log dict contains pairs of all items to log, which includes
            percentage complete and projected time left if total > 0
        """
        # from parlai.core.metrics import Metric  # delay import to prevent circular dep

        if isinstance(done, Metric):
            done = done.value()
        self.tot_time += self.timer.time()
        self.timer.reset()
        if report:
            report['exs'] = done
        if total > 0 and done > 0:
            progress = done / total
            seconds_left = max(0, self.tot_time / progress - self.tot_time)
            eta = timedelta(seconds=int(seconds_left + 0.5))
        else:
            progress = 0
            eta = "unknown"
        elapsed = timedelta(seconds=int(self.tot_time))

        text = (
            f'{progress:.1%} complete ({done:,d} / {total:,d}), '
            f'{elapsed} elapsed, {eta} eta'
        )
        if report:
            report_s = nice_report(report)
            text = f'{text}\n{report_s}'
        return text, report


class AttrDict(dict):
    """
    Helper class to have a dict-like object with dot access.

    For example, instead of `d = {'key': 'value'}` use
    `d = AttrDict(key='value')`.
    To access keys, instead of doing `d['key']` use `d.key`.

    While this has some limitations on the possible keys (for example, do not
    set the key `items` or you will lose access to the `items()` method), this
    can make some code more clear.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize AttrDict using input dict.
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class SimpleCounter:
    """
    Simple counter object.
    """

    def __init__(self, value=0):
        self.val = value

    def increment(self, value=1):
        self.val += value

    def value(self):
        return self.val


def _report_sort_key(report_key: str) -> Tuple[str, str]:
    """
    Sorting name for reports.

    Sorts by main metric alphabetically, then by task.
    """
    # if metric is on its own, like "f1", we will return ('', 'f1')
    # if metric is from multitask, we denote it.
    # e.g. "convai2/f1" -> ('convai2', 'f1')
    # we handle multiple cases of / because sometimes teacher IDs have
    # filenames.
    fields = report_key.split("/")
    main_key = fields.pop(-1)
    sub_key = '/'.join(fields)
    return (sub_key or 'all', main_key)


def float_formatter(f: Union[float, int]) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def _line_width():
    if os.environ.get('PARLAI_FORCE_WIDTH'):
        try:
            return int(os.environ['PARLAI_FORCE_WIDTH'])
        except ValueError:
            pass
    try:
        # if we're in an interactive ipython notebook, hardcode a longer width
        __IPYTHON__
        return 128
    except NameError:
        return shutil.get_terminal_size((88, 24)).columns


def nice_report(report) -> str:
    """
    Render an agent Report as a beautiful string.

    If pandas is installed,  we will use it to render as a table. Multitask
    metrics will be shown per row, e.g.

    .. code-block:
                 f1   ppl
       all     .410  27.0
       task1   .400  32.0
       task2   .420  22.0

    If pandas is not available, we will use a dict with like-metrics placed
    next to each other.
    """
    if not report:
        return ""

    # from parlai.core.metrics import Metric

    try:
        import pandas as pd

        use_pandas = True
    except ImportError:
        use_pandas = False

    sorted_keys = sorted(report.keys(), key=_report_sort_key)
    output: OrderedDict[Union[str, Tuple[str, str]], float] = OrderedDict()
    for k in sorted_keys:
        v = report[k]
        if isinstance(v, Metric):
            v = v.value()
        if use_pandas:
            output[_report_sort_key(k)] = v
        else:
            output[k] = v

    if use_pandas:
        line_width = _line_width()

        df = pd.DataFrame([output])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.stack().transpose().droplevel(0, axis=1)
        result = "   " + df.to_string(
            na_rep="",
            line_width=line_width - 3,  # -3 for the extra spaces we add
            float_format=float_formatter,
            index=df.shape[0] > 1,
        ).replace("\n\n", "\n").replace("\n", "\n   ")
        result = re.sub(r"\s+$", "", result)
        return result
    else:
        return json.dumps(
            {
                k: round_sigfigs(v, 4) if isinstance(v, float) else v
                for k, v in output.items()
            }
        )


def round_sigfigs(x: Union[float, 'torch.Tensor'], sigfigs=4) -> float:
    """
    Round value to specified significant figures.

    :param x: input number
    :param sigfigs: number of significant figures to return

    :returns: float number rounded to specified sigfigs
    """
    x_: float
    if __TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x_ = x.item()
    else:
        x_ = x  # type: ignore

    try:
        if x_ == 0:
            return 0
        return round(x_, -(math.floor(math.log10(abs(x_)) - sigfigs + 1)))
    except (ValueError, OverflowError) as ex:
        if x_ in [float('inf'), float('-inf')] or x_ != x_:  # inf or nan
            return x_
        else:
            raise ex


def clip_text(text, max_len):
    """
    Clip text to max length, adding ellipses.
    """
    if len(text) > max_len:
        begin_text = ' '.join(text[: math.floor(0.8 * max_len)].split(' ')[:-1])
        end_text = ' '.join(
            text[(len(text) - math.floor(0.2 * max_len)) :].split(' ')[1:]
        )
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text


def _ellipse(lst: List[str], max_display: int = 5, sep: str = '|') -> str:
    """
    Like join, but possibly inserts an ellipsis.

    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '... ({} of {} shown)'.format(max_display, len(choices))
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def display_messages(
    msgs: List[Dict[str, Any]],
    prettify: bool = False,
    ignore_agent_reply: bool = False,
    add_fields: str = '',
    max_len: int = 1000,
    verbose: bool = False,
) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. add_fields provides
    a list of fields in the msgs which should be displayed if verbose is off.
    """

    def _token_losses_line(
        msg: Dict[str, Any], fields_to_show: List[str], space: str
    ) -> Optional[str]:
        """
        Displays the loss associated with each token. Can be used for debugging
        generative models.

        See TorchGeneratorAgent._construct_token_losses for an example implementation.
        """
        key = 'token_losses'
        token_losses = msg.get(key, None)
        if key not in fields_to_show or not token_losses:
            return None
        # Reduce losses to 4 significant figures
        formatted_tl = ' | '.join(
            [f"{tl[0]} {float('{:.4g}'.format(tl[1]))}" for tl in token_losses]
        )
        return _pretty_lines(space, key, formatted_tl, 'text2')

    def _pretty_lines(indent_space, field, value, style):
        line = '{}{} {}'.format(
            indent_space, colorize('[' + field + ']:', 'field'), colorize(value, style)
        )
        return line

    lines = []
    episode_done = False
    extra_add_fields_ = add_fields.split(',')
    for index, msg in enumerate(msgs):
        if msg is None or (index == 1 and ignore_agent_reply):
            # We only display the first agent (typically the teacher) if we
            # are ignoring the agent reply.
            continue

        if msg.get('episode_done'):
            episode_done = True
        # Possibly indent the text (for the second speaker, if two).
        space = ''
        if len(msgs) == 2 and index == 1:
            space = '   '

        agent_id = msg.get('id', '[no id field]')
        if verbose:
            line = _pretty_lines(
                indent_space=space, field='id', value=agent_id, style='id'
            )
            lines.append(line)

        # Only display rewards !=0 as they are confusing in non-RL tasks.
        if msg.get('reward', 0) != 0:
            lines.append(space + '[reward: {r}]'.format(r=msg['reward']))

        fields_to_show = []
        if verbose:
            fields_to_show = [field for field in msg]
        else:
            fields_to_show = [
                field
                for field in msg
                if field in list(MUST_SHOW_MESSAGE_FIELDS) + extra_add_fields_
            ]
        fields_to_show.sort()

        # Display fields without special format
        for field in fields_to_show:
            if field not in SPECIAL_FORMATED_DISPLAY_MESSAGE_FIELDS:
                if type(msg[field]) is list:
                    value = _ellipse(msg[field], sep='\n  ')
                else:
                    value = clip_text(str(msg.get(field)), max_len)
                line = _pretty_lines(
                    indent_space=space, field=field, value=value, style='text2'
                )
                lines.append(line)

        # Display fields WITH special format requirements
        # Display Image
        if type(msg.get('image')) in [str, torch.Tensor]:
            lines.append(f'[ image ]: {msg["image"]}')
        # Display Text
        if msg.get('text', ''):
            value = clip_text(msg['text'], max_len)
            style = 'bold_text' if index == 0 else 'labels'
            field = 'text' if verbose else agent_id
            line = _pretty_lines(
                indent_space=space, field=field, value=value, style=style
            )
            lines.append(line)
        # Display Label Fields
        for field in {'labels', 'eval_labels', 'label_candidates', 'text_candidates'}:
            if msg.get(field) and field in fields_to_show:
                line = _pretty_lines(
                    indent_space=space,
                    field=field,
                    value=_ellipse(msg[field]),
                    style=field,
                )
                lines.append(line)
        if msg.get('metrics') and verbose:
            lines.append(
                _pretty_lines(
                    indent_space=space,
                    field='metrics',
                    value="\n" + nice_report(msg['metrics']),
                    style='text',
                )
            )

        # Handling this separately since we need to clean up the raw output before displaying.
        token_loss_line = _token_losses_line(msg, fields_to_show, space)
        if token_loss_line:
            lines.append(token_loss_line)

    if episode_done:
        lines.append(
            colorize('- - - - - - - END OF EPISODE - - - - - - - - - -', 'highlight')
        )

    return '\n'.join(lines)


def str_to_msg(txt, ignore_fields=''):
    """
    Convert formatted string to ParlAI message dict.

    :param txt:
        formatted string to convert. String format is tab-separated fields,
        with colon separating field name and contents.
    :param ignore_fields:
        (default '') comma-separated field names to not
        include in the msg dict even if they're in the string.
    """

    def tostr(txt):
        txt = str(txt)
        txt = txt.replace('\\t', '\t')
        txt = txt.replace('\\n', '\n')
        txt = txt.replace('__PIPE__', '|')
        return txt

    def tolist(txt):
        vals = txt.split('|')
        for i, v in enumerate(vals):
            v = tostr(v)
            vals[i] = v
        return vals

    def convert(key, value):
        if key == 'text' or key == 'id':
            return tostr(value)
        elif (
            key == 'label_candidates'
            or key == 'labels'
            or key == 'eval_labels'
            or key == 'text_candidates'
        ):
            return tolist(value)
        elif key == 'reward':
            try:
                return int(value)
            except ValueError:
                return float(value)
        elif key == 'episode_done':
            return bool(value)
        else:
            return tostr(value)

    if txt == '' or txt is None:
        return None

    msg = {}
    for t in txt.split('\t'):
        ind = t.find(':')
        key = t[:ind]
        value = t[ind + 1 :]
        if key not in ignore_fields.split(','):
            msg[key] = convert(key, value)
    msg['episode_done'] = msg.get('episode_done', False)
    return Message(msg)


def msg_to_str(msg, ignore_fields=''):
    """
    Convert ParlAI message dict to string.

    :param msg:
        dict to convert into a string.
    :param ignore_fields:
        (default '') comma-separated field names to not include in the string
        even if they're in the msg dict.
    """

    def filter(txt):
        txt = str(txt)
        txt = txt.replace('\t', '\\t')
        txt = txt.replace('\n', '\\n')
        txt = txt.replace('|', '__PIPE__')
        return txt

    def add_field(name, data):
        if name == 'reward' and data == 0:
            return ''
        if name == 'episode_done' and data is False:
            return ''
        txt = ''
        if type(data) == tuple or type(data) == set or type(data) == list:
            # list entries
            for c in data:
                txt += filter(c) + "|"
            txt = txt[:-1]
        else:
            # single fields
            txt = filter(data)
        return name + ":" + txt + '\t'

    default_fields = [
        'id',
        'text',
        'labels',
        'label_candidates',
        'episode_done',
        'reward',
    ]
    txt = ""
    ignore_fields = ignore_fields.split(',')
    for f in default_fields:
        if f in msg and f not in ignore_fields:
            txt += add_field(f, msg[f])
    for f in msg.keys():
        if f not in default_fields and f not in ignore_fields:
            txt += add_field(f, msg[f])
    return txt.rstrip('\t')


# DEPRECATION DAY: DELETE
def set_namedtuple_defaults(namedtuple, default=None):
    """
    Set *all* of the fields for a given nametuple to a singular value.

    Additionally removes the default docstring for each field.
    Modifies the tuple in place, but returns it anyway.

    More info:
    https://stackoverflow.com/a/18348004

    :param namedtuple: A constructed collections.namedtuple
    :param default: The default value to set.

    :returns: the modified namedtuple
    """
    namedtuple.__new__.__defaults__ = (default,) * len(namedtuple._fields)
    for f in namedtuple._fields:
        del getattr(namedtuple, f).__doc__
    return namedtuple


_seen_logs: Set[str] = set()


def warn_once(msg: str) -> None:
    """
    Log a warning, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.warning(msg)


def error_once(msg: str) -> None:
    """
    Log an error, but only once.

    :param str msg: Message to display
    """
    global _seen_logs
    if msg not in _seen_logs:
        _seen_logs.add(msg)
        logging.error(msg)


def recursive_getattr(obj, attr, *args):
    """
    Recursive call to getattr for nested attributes.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

# import parlai.utils.logging as logging

BAR = '=' * 60
SMALL_BAR = '-' * 60


class Metadata:
    """
    Utility class for conversation metadata.

    Metadata should be saved at ``<datapath>.metadata``.
    """

    def __init__(self, datapath):
        self._load(datapath)

    def _load(self, datapath):
        self.metadata_path = self._get_path(datapath)
        if not PathManager.exists(self.metadata_path):
            raise RuntimeError(
                f'Metadata at path {self.metadata_path} not found. '
                'Double check your path.'
            )

        with PathManager.open(self.metadata_path, 'rb') as f:
            metadata = json.load(f)

        self.datetime = metadata['date']
        self.opt = metadata['opt']
        self.self_chat = metadata['self_chat']
        self.speakers = metadata['speakers']
        self.version_num = metadata['version']
        self.extra_data = {}
        for k, v in metadata.items():
            if k not in ['date', 'opt', 'speakers', 'self_chat', 'version']:
                self.extra_data[k] = v

    def read(self):
        """
        Read the relevant metadata.
        """
        string = f'Metadata version {self.version_num}\n'
        string += f'Saved at: {self.datetime}\n'
        string += f'Self chat: {self.self_chat}\n'
        string += f'Speakers: {self.speakers}\n'
        string += 'Opt:\n'
        for k, v in self.opt.items():
            string += f'\t{k}: {v}\n'
        for k, v in self.extra_data.items():
            string += f'{k}: {v}\n'

        return string

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.metadata'

    @staticmethod
    def version():
        return '0.1'

    @classmethod
    def save_metadata(cls, datapath, opt, self_chat=False, speakers=None, **kwargs):
        """
        Dump conversation metadata to file.
        """
        metadata = {}
        metadata['date'] = str(datetime.datetime.now())
        metadata['opt'] = opt
        metadata['self_chat'] = self_chat
        metadata['speakers'] = speakers
        metadata['version'] = cls.version()

        for k, v in kwargs.items():
            metadata[k] = v

        metadata_path = cls._get_path(datapath)
        logging.info(f'Writing metadata to file {metadata_path}')
        with PathManager.open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata))


class Turn(AttrDict):
    """
    Utility class for a dialog turn.
    """

    def __init__(self, id=None, text=None, **kwargs):
        super().__init__(self, id=id, text=text, **kwargs)


class Conversation:
    """
    Utility class for iterating through a single episode.

    Used in the context of the Conversations class.
    """

    def __init__(self, episode):
        self.episode = episode
        self.context = episode.get('context')
        self.metadata_path = episode.get('metadata_path')
        self.turns = self._build_turns(episode)

    def _build_turns(self, episode):
        turns = []
        for act_pair in episode['dialog']:
            for act in act_pair:
                turns.append(Turn(**act))
        return turns

    def __str__(self):
        string = BAR + '\n'
        high_level = [k for k in self.episode.keys() if k != 'dialog']
        if high_level:
            for key in high_level:
                string += f'{key}: {self.episode[key]}\n'
            string += SMALL_BAR + '\n'

        for turn in self.turns:
            string += f'{turn.id}: {turn.text}\n'

        string += BAR + '\n'
        return string

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, index):
        return self.turns[index]

    def __iter__(self):
        self.iterator_idx = 0
        return self

    def __next__(self):
        """
        Return the next conversation.
        """
        if self.iterator_idx >= len(self.turns):
            raise StopIteration

        conv = self.turns[self.iterator_idx]
        self.iterator_idx += 1

        return conv


class Conversations:
    """
    Utility class for reading and writing from ParlAI Conversations format.

    Conversations should be saved in JSONL format, where each line is
    a JSON of the following form:

    WARNING: The data below must be on ONE LINE per dialogue
    in a conversation file or it will not load!!

    .. code-block:

        {
            'possible_conversation_level_info': True,
            'dialog':
                [   [
                        {
                            'id': 'speaker_1',
                            'text': <first utterance>,
                        },
                        {
                            'id': 'speaker_2',
                            'text': <second utterance>,
                        },
                        ...
                    ],
                    ...
                ]
            ...
        }
    """

    def __init__(self, datapath):
        self._datapath = datapath
        self.metadata = self._load_metadata(datapath)

    def __len__(self):
        return sum(1 for _ in self._load_raw(self._datapath))

    def _load_raw(self, datapath):
        """
        Load the data as a raw, unparsed file.

        Useful for fast IO stuff like random indexing.
        """
        if not PathManager.exists(datapath):
            raise RuntimeError(
                f'Conversations at path {datapath} not found. '
                'Double check your path.'
            )

        with PathManager.open(datapath, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                yield line

    def _parse(self, line):
        return Conversation(json.loads(line))

    def _load_conversations(self, datapath):
        return (self._parse(line) for line in self._load_raw(datapath))

    def _load_metadata(self, datapath):
        """
        Load metadata.

        Metadata should be saved at <identifier>.metadata
        Metadata should be of the following format:
        {
            'date': <date collected>,
            'opt': <opt used to collect the data>,
            'speakers': <identity of speakers>,
            ...
            Other arguments.
        }
        """
        try:
            metadata = Metadata(datapath)
            return metadata
        except RuntimeError:
            logging.debug('Metadata does not exist. Please double check your datapath.')
            return None

    def read_metadata(self):
        if self.metadata is not None:
            logging.info(self.metadata)
        else:
            logging.warning('No metadata available.')

    def __getitem__(self, index):
        raw = self._load_raw(self._datapath)
        item = list(itertools.islice(raw, index, index + 1))[0]
        return self._parse(item)

    def __iter__(self):
        return self._load_conversations(self._datapath)

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.jsonl'

    @staticmethod
    def _check_parent_dir_exits(datapath):
        parent_dir = os.path.dirname(datapath)
        if not parent_dir or PathManager.exists(parent_dir):
            return
        logging.info(f'Parent directory ({parent_dir}) did not exist and was created.')
        PathManager.mkdirs(parent_dir)

    @classmethod
    def save_conversations(
        cls,
        act_list,
        datapath,
        opt,
        save_keys='all',
        context_ids='context',
        self_chat=False,
        **kwargs,
    ):
        """
        Write Conversations to file from an act list.

        Conversations assume the act list is of the following form: a list of episodes,
        each of which is comprised of a list of act pairs (i.e. a list dictionaries
        returned from one parley)
        """
        cls._check_parent_dir_exits(datapath)
        to_save = cls._get_path(datapath)

        context_ids = context_ids.strip().split(',')
        # save conversations
        speakers = []
        with PathManager.open(to_save, 'w') as f:
            for ep in act_list:
                if not ep:
                    continue
                convo = {
                    'dialog': [],
                    'context': [],
                    'metadata_path': Metadata._get_path(to_save),
                }
                for act_pair in ep:
                    new_pair = []
                    for ex in act_pair:
                        ex_id = ex.get('id')
                        if ex_id in context_ids:
                            context = True
                        else:
                            context = False
                            if ex_id not in speakers:
                                speakers.append(ex_id)

                        # set turn
                        turn = {}
                        if save_keys != 'all':
                            save_keys_lst = save_keys.split(',')
                        else:
                            save_keys_lst = ex.keys()
                        for key in save_keys_lst:
                            turn[key] = ex.get(key, '')
                            if key == 'metrics':
                                turn[key] = dict_report(turn[key])
                        turn['id'] = ex_id
                        if not context:
                            new_pair.append(turn)
                        else:
                            convo['context'].append(turn)
                    if new_pair:
                        convo['dialog'].append(new_pair)
                json_convo = json.dumps(convo, default=lambda v: '<not serializable>')
                f.write(json_convo + '\n')
        logging.info(f'Conversations saved to file: {to_save}')

        # save metadata
        Metadata.save_metadata(
            to_save, opt, self_chat=self_chat, speakers=speakers, **kwargs
        )


# from parlai.utils.data import DatatypeHelper
# from parlai.utils.misc import AttrDict, str_to_msg, warn_once, SimpleCounter
# from parlai.utils.distributed import get_rank, num_workers
# from parlai.utils.distributed import is_distributed

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Useful utilities for training in distributed mode.

Many of these functions act as wrappers which perform no-ops if code is running in non-
distributed mode.
"""

import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
# import parlai.utils.logging as logging

try:
    import torch.nn
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_distributed():
    """
    Return if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def num_workers():
    """
    Get the total number of workers.
    """
    if not is_distributed():
        return 1
    else:
        return dist.get_world_size()


def is_primary_worker():
    """
    Determine if we are the primary (rank 0)  worker.

    Returns False if we are a secondary worker. Returns True if we are either (1) not in
    distributed mode (2) or are the primary (rank 0) worker.
    """
    return not is_distributed() or dist.get_rank() == 0


def get_rank():
    """
    Returns the rank of the current worker.

    Returns 0 if not in distributed.
    """
    if not is_distributed():
        return 0
    else:
        return dist.get_rank()


@contextlib.contextmanager
def override_print(suppress=False, prefix=None):
    """
    Context manager to override the print to suppress or modify output.

    Recommended usage is to call this with suppress=True for all non-primary
    workers, or call with a
    prefix of rank on all workers.

    >>> with override_print(prefix="rank{}".format(rank)):
    ...     my_computation()
    :param bool suppress:
        if true, all future print statements are noops.
    :param str prefix:
        if not None, this string is prefixed to all future print statements.
    """
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        if suppress:
            # do nothing
            return
        elif prefix:
            return builtin_print(prefix, *args, **kwargs)
        else:
            # default to normal print
            return builtin_print(*args, **kwargs)

    if prefix:
        logging.logger.add_format_prefix(prefix)
    if suppress:
        logging.disable()

    # override the print for now
    builtins.print = new_print
    yield
    # bring it back at the end of the context
    builtins.print = builtin_print

    if suppress:
        logging.enable()


def all_gather_list(data):
    """
    Gather arbitrary data from all nodes into a list.

    Similar to `~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    :param data:
        data from the local worker to be gathered on other workers

    :returns:
        a list containing [data1, data2, ...] of all workers
    """
    if not is_distributed():
        # fall back to just keeping things basic if we're not distributed
        return [data]

    # stolen shamelessly from fairseq
    # https://github.com/pytorch/fairseq/blob/c37250ab1c845919af721cd3f5c4cec2993aefe1/fairseq/distributed_utils.py#L116-L170
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    enc = list(pickle.dumps(data))
    enc_size = len(enc)

    # find the sizes of all the serialized items
    sizes = torch.zeros(world_size, dtype=torch.long).cuda()
    sizes[rank] = enc_size
    dist.all_reduce(sizes)

    # need to know our positions
    sizes = sizes.cpu()
    positions = sizes.cumsum(dim=0)

    buffer_size = positions[-1].item()
    buffer = torch.cuda.ByteTensor(buffer_size).zero_()

    start = positions[rank] - enc_size
    end = positions[rank]
    buffer[start:end] = torch.ByteTensor(enc)

    dist.all_reduce(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer[positions[i] - sizes[i] : positions[i]]
        try:
            result.append(pickle.loads(bytes(out_buffer.tolist())))
        except pickle.UnpicklingError:
            raise RuntimeError(
                'There was an unpickling error in all_gather_list. This likely '
                'means your workers got out of synchronization (e.g. one is '
                'expecting to sync and another is not.)'
            )

    return result


def sync_object(data):
    """
    Sync an object among all workers.

    All workers will return the same value for `data` when returning from this
    method, always using the primary worker's version. Useful for ensuring control
    flow decisions are made the same.

    :param object data:
        The object to synchronize. Must be pickleable.

    :return: the synchronized data
    """
    value = all_gather_list(data if get_rank() == 0 else None)[0]
    return value


def sync_parameters(model: torch.nn.Module) -> bool:
    """
    Sync all parameters across all workers are the same.

    Always returns True, or raises an AssertionError if there was a failure.

    :param model: A pytorch model.
    :return: always True
    """
    if not is_distributed():
        # if things aren't distributed, of course things are in sync
        return True

    # sync all the parameters
    with torch.no_grad():
        for p in model.parameters():
            if not is_primary_worker():
                # zero out parameters on all workers EXCEPT the primary worker
                p.data.zero_()
            # sum the parameters across all workers, resulting in everyone having
            # the parameters of the primary worker
            dist.all_reduce(p.data, dist.ReduceOp.SUM)

    # double check everything synced correctly
    norm2 = sum((p.data**2).sum().float().item() for p in model.parameters())
    all_versions = all_gather_list(norm2)
    if not all(n == norm2 for n in all_versions):
        raise AssertionError(
            "Some models parameters were out of sync. Got the following norms: {}".format(
                " ".join(str(x) for x in all_versions)
            )
        )

    return True


@contextlib.contextmanager
def distributed_context(
    rank, opt, rank_offset=0, gpu=None, init_method="tcp://localhost:61337"
):
    """
    A context which wraps initialization of a distributed/multiprocessing run.

    Every process in the distributed run should launch with this. In true
    distributed setting you may wish to use slurm_distributed_context instead.

    :param int rank:
        This process's rank, less rank_offset.
    :param int rank_offset:
        Used as an offset of rank. Used between multiprocessing vs true distributed,
        and a hack around torch.multiprocessing.spawn being only used for the
        non-primary workers.
    :param opt:
        command line options
        distributed training setups on the same machine.
    :param int gpu:
        Which GPU to use. Defaults to using rank and local devices, but must be
        manually specified when using many-hosts.
    :param str init method:
        Init method, such as ``tcp://localhost:61337``. See torch.distributed docs.
    """
    # Set per-host options
    opt = copy.deepcopy(opt)
    # we need to manually adjust the rank differently in multiprocessing
    # and distributed train
    rank = rank + rank_offset
    opt['rank'] = rank
    if gpu is None:
        # default assumption is local GPUs
        gpu = rank % torch.cuda.device_count()
    opt['gpu'] = gpu
    # make sure we don't just use whatever GPU was saved in the model file
    if 'override' not in opt:
        opt['override'] = {}
    opt['override']['gpu'] = gpu

    # Suppress output of workers except the main host.
    if opt.get('verbose') or rank != 0:
        print_prefix = 'rank:{:3d} |'.format(rank)
    else:
        print_prefix = None
    suppress_output = not opt.get('verbose') and rank != 0

    with override_print(suppress_output, print_prefix):
        # perform distributed setup, ensuring all hosts are ready
        if opt['gpu'] != -1:
            torch.cuda.set_device(opt['gpu'])
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=opt['distributed_world_size'],
            rank=rank,
        )
        logging.info("Distributed group initialized")

        # manual_seed can be a noop without this
        torch.cuda.init()
        # make sure all parameters will be in sync
        torch.manual_seed(42)
        # force a sync so that no one gets ahead, and all are seeded together
        sync_object(None)

        try:
            yield opt
        finally:
            dist.destroy_process_group()


def get_dist_group():
    """
    Find the default pytorch distributed group.

    Used within FSDP to mark which workers are participating. Important to manually call
    this because FSDP will cache old groups, but our test suite will instantiate new
    groups per test.
    """
    from torch.distributed.distributed_c10d import _get_default_group

    return _get_default_group()


@contextlib.contextmanager
def slurm_distributed_context(opt):
    """
    Initialize a distributed context, using the SLURM environment.

    Does some work to read the environment to find a list of participating nodes
    and the main node.

    :param opt:
        Command line options.
    """
    # We can determine the init method automatically for Slurm.
    # double check we're using SLURM
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    if node_list is None:
        raise RuntimeError(
            'Does not appear to be in a SLURM environment. '
            'You should not call this script directly; see launch_distributed.py'
        )
    try:
        # Figure out the main host, and which rank we are.
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]
        )
    except FileNotFoundError as e:
        # Slurm is not installed
        raise RuntimeError(
            f'SLURM does not appear to be installed. Missing file: {e.filename}'
        )

    main_host = hostnames.split()[0].decode('utf-8')
    distributed_rank = int(os.environ['SLURM_PROCID'])
    if opt.get('model_parallel'):
        # -1 signals to multiprocessing_train to use all GPUs available.
        # (A value of None signals to multiprocessing_train to use the GPU
        # corresponding to the rank.
        device_id = -1
    else:
        device_id = int(os.environ['SLURM_LOCALID'])
    port = opt['port']
    logging.info(
        f'Initializing host {socket.gethostname()} as rank {distributed_rank}, '
        f'main is {main_host}'
    )
    # Begin distributed training
    with distributed_context(
        distributed_rank, opt, 0, device_id, init_method=f"tcp://{main_host}:{port}"
    ) as opt:
        yield opt


def find_free_port() -> int:
    """
    Find a free port we can bind to locally.

    Credit: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# import parlai.utils.torch as torch_utils

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from iopath.common.file_io import PathManager as _PathManager
except ImportError:
    try:
        from fvcore.common.file_io import PathManagerBase as _PathManager
    except ImportError:
        raise ImportError(
            "parlai now requires iopath for some I/O operations. Please run "
            "`pip install iopath`"
        )

USE_ATOMIC_TORCH_SAVE = True

PathManager = _PathManager()

try:
    # register any internal file handlers
    import parlai_fb  # noqa: F401

    # internal file handlers can't handle atomic saving. see T71772714
    USE_ATOMIC_TORCH_SAVE = not parlai_fb.finalize_registration(PathManager)
except ModuleNotFoundError:
    USE_ATOMIC_TORCH_SAVE = True


def atomic_save(state_dict: Any, path: str) -> None:
    """
    Like torch.save, but atomic.

    Useful for preventing trouble coming from being pre-empted or killed while writing
    to disk. Works by writing to a temporary file, and then renaming the file to the
    final name.
    """

    if USE_ATOMIC_TORCH_SAVE:
        with open(path + ".tmp", "wb") as f:
            torch.save(state_dict, f)
        os.replace(path + ".tmp", path)
    else:
        with PathManager.open(path, "wb") as f:
            torch.save(state_dict, f)

# import parlai.utils.logging as logging
# from parlai.utils.io import PathManager
# from parlai.core.mutators import Mutator

#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from __future__ import annotations

import abc
import importlib
import pkgutil
from typing import Optional, Iterable, Tuple, List, Iterator, Callable, Type

# import parlai.mutators
# from parlai.core.params import ParlaiParser
# from parlai.core.opt import Opt
# from parlai.core.message import Message

MUTATOR_REGISTRY: dict[str, Type] = {}


def setup_mutator_registry():
    """
    Loads the mutators so that @register_mutator is hit for all.
    """
    global MUTATOR_REGISTRY
    if hasattr(setup_mutator_registry, 'loaded'):
        return
    # for module in pkgutil.iter_modules(parlai.mutators.__path__, 'parlai.mutators.'):
    #     importlib.import_module(module.name)
    try:
        import parlai_fb.mutators

        for module in pkgutil.iter_modules(
            parlai_fb.mutators.__path__, 'parlai_fb.mutators.'
        ):
            importlib.import_module(module.name)
    except ImportError:
        pass
    try:
        import parlai_internal.mutators

        for module in pkgutil.iter_modules(
            parlai_internal.mutators.__path__, 'parlai_internal.mutators.'
        ):
            importlib.import_module(module.name)
    except ImportError:
        pass
    setup_mutator_registry.loaded = True
    return MUTATOR_REGISTRY


def register_mutator(name: str) -> Callable[[Type], Type]:
    """
    Register a mutator.
    """

    def _inner(cls_: Type) -> Type:
        global MUTATOR_REGISTRY
        if name in MUTATOR_REGISTRY and cls_ is not MUTATOR_REGISTRY[name]:
            raise NameError(
                "Mutators must be uniquely named, but detected two mutators with "
                f"the name '{name}'."
            )
        MUTATOR_REGISTRY[name] = cls_
        return cls_

    return _inner


class Mutator(abc.ABC):
    """
    Base class for mutators.

    Users are not advised to use this class.
    """

    @classmethod
    def load_mutator_types(cls, mutator_names: Optional[str]) -> List[Type]:
        """
        Map mutator names to actual classes via the registry.

        :param mutator_names:
            A list of one or more mutators separated by '+'. E.g.
            'flatten+word_shuffle'.
        :returns: a list of mutators
        """

        global MUTATOR_REGISTRY
        setup_mutator_registry()
        if not mutator_names:
            return []
        assert isinstance(mutator_names, str)
        names = mutator_names.replace('+', ',').split(',')
        mutators = [MUTATOR_REGISTRY[name] for name in names]
        return mutators

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        pass

    def __init__(self, opt):
        self.opt = opt

    def _pop_episode_done(self, message: Message) -> Tuple[Message, bool]:
        try:
            episode_done = message.pop('episode_done')
        except KeyError:
            episode_done = False
        return message, episode_done

    def _group_into_episodes(
        self, message_stream: Iterable[Message]
    ) -> Iterator[List[Message]]:
        """
        Apply fn to grouped episodes, yielding back the results of the application.
        """
        episode: List[Message] = []
        for message in message_stream:
            if message.is_padding():
                assert not episode
                yield [message]
                continue
            message, episode_done = self._pop_episode_done(message)
            episode.append(message)
            if episode_done:
                yield episode
                episode = []
        if episode:
            yield episode

    def _add_episode_done(self, episode: List[Message]) -> List[Message]:
        for i, message in enumerate(episode):
            message['episode_done'] = i == len(episode) - 1
        return episode

    @abc.abstractmethod
    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        pass


class MessageMutator(Mutator):
    """
    Message-level mutators.

    Message-level mutators have a function applied per-utterance. They are ideal
    for transformations of data which don't create any new conversations or
    turns, but only apply simple text-transformations.

    Examples include:

    * Shuffling words in context
    * Adding a special token based on a non-text field
    * Replacing words with synonyms or other simple augmentations
    """

    @abc.abstractmethod
    def message_mutation(self, message: Message) -> Message:
        """
        Abstract message mutation.

        The main method to implement when implementing an MessageMutator.

        :param message:
            An individual message you should mutate.
        :returns:
            The mutated message.
        """
        pass

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        for message in messages:
            if message.is_padding():
                yield message
                continue
            message, episode_done = self._pop_episode_done(message)
            message = self.message_mutation(message)
            if 'episode_done' in message:
                raise ValueError('MessageMutators should not modify episode_done.')
            message['episode_done'] = episode_done
            yield message


class EpisodeMutator(Mutator):
    """
    Episode-level mutators.
    """

    @abc.abstractmethod
    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        """
        Abstract episode mutation.

        The main method to implement when implementing an EpisodeMutator.

        The "episode_done" field will be automatically stripped before providing
        as input, and automatically added back to the finalized episode.

        :param messages:
            All the messages in one episode. You may manipulate any or all of
            them, or change the ordering entirely.
        :returns:
            The new, mutated episode.
        """
        pass

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """
        for episode in self._group_into_episodes(messages):
            if episode and episode[0].is_padding():
                for message in episode:
                    yield message
            else:
                mutated_episode = self._add_episode_done(self.episode_mutation(episode))
                yield from mutated_episode


class ManyEpisodeMutator(Mutator):
    """
    Episode mutator than can map one episode to zero or more.
    """

    @abc.abstractmethod
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        """
        Abstract many-episode mutation.

        The main method to implement when creation a ManyEpisodeMutator.
        You should map this episode to zero-or-more episodes.

        If you wish to create multiple episodes, you need to output
        one-sublist-per-new-episode. As with EpisodeMutator, "episode_done"
        will be automatically stripped and re-inserted for you.

        :param episode:
            A single episode (provided list of Messages).
        :returns:
            A list of list of messages. Each sub-list will be turned into a new
            episode.
        """
        pass

    def __call__(self, messages: Iterable[Message]) -> Iterator[Message]:
        """
        Apply the mutator to a series of messages.

        Not meant to be called directly by a user.
        """

        for episode in self._group_into_episodes(messages):
            if episode and episode[0].is_padding():
                yield from episode
            else:
                mutated_episodes = self.many_episode_mutation(episode)
                for mutated_episode in mutated_episodes:
                    yield from self._add_episode_done(mutated_episode)



from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import concurrent.futures
import copy
import json
import os
import queue
import random
import yaml
from threading import Thread
import torch
from typing import List, Tuple, Optional, TypeVar, Any


ERROR_MESSAGE_NO_DATAFILE = (
    "{class_name} is expected to set self.opt['datafile'] inside `__init__` "
    "before calling `super().__init__`. This will passed to setup_data, "
    "indicating what data to load. If you don't know what to use, set "
    "`opt['datafile'] = parlai.utils.data.DatatypeHelper.fold(opt['datatype'])` "
    "to receive the fold name in setup_data."
)


ChunkOutput = TypeVar('ChunkOutput')


class DataLoader(Thread):
    """
    A worker thread that provides a threadpool for data loading.

    A teacher may submit a request to the loader, which will return the
    appropriate data.

    To submit a request, a teacher should call ``request_load``.
    """

    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_workers = opt.get('num_load_threads', 1)
        self.request_queue = queue.Queue()
        self.last_future = None

    def request_load(self, receive_fn, load_fn, args):
        """
        Queue a request for loading.

        :param receive_fn:
            a receive function (for receiving the data)
        :param load_fn:
            a load function (for loading the data)
        :param args:
            arguments for the load function. args can be either a dictionary of
            arguments for a function, or a list of positional arguments
        """
        self.request_queue.put((receive_fn, load_fn, args))

    def run(self):
        """
        Run the execution loop.
        """
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix=self.name
        )
        with executor:
            while True:
                receive_fn, load_fn, args = self.request_queue.get()
                if receive_fn is StopIteration:
                    return
                try:
                    if type(args) == dict:
                        future = executor.submit(load_fn, **args)
                    else:
                        future = executor.submit(load_fn, *args)
                    self.last_future = future
                    receive_fn(future)
                except RuntimeError:
                    return


class _ErrorThrowingDataLoader(object):
    """
    A fake DataLoader which throws an exception when a work order is placed.

    Since threads cannot be mixed with spawn_method='fork', we need to disallow users
    from combining --num-workers with teachers that utilize threads. This placeholder
    object is only useful for ensuring the user sees a loud error message when they
    accidentally use a thread.
    """

    def __init__(self, opt):
        pass

    def request_load(self, receive_fn, load_fn, args):
        raise RuntimeError(
            'One of your teachers uses a DataLoader or a thread. You may only '
            'combine this with --num-workers 0.'
        )

    def start(self):
        pass


class Teacher(Agent):
    """
    Basic Teacher agent that keeps track of how many times it's received messages.

    Teachers provide the ``report()`` method to get back metrics.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument('--teacher_seed', default=None, type=float)
        parser.add_argument(
            '--mutators',
            '-mut',
            default=None,
            help='Apply one or more mutators to the data.',
        )
        mutators = Mutator.load_mutator_types(
            partial_opt.get('mutators') if partial_opt else None
        )
        for m in mutators:
            m.add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        if not hasattr(self, 'id'):
            self.id = opt.get('task', 'teacher')
        if not hasattr(self, 'metrics'):
            self.metrics = TeacherMetrics(
                metrics_list=opt.get('metrics', 'default'),
                shared=shared['metrics'] if shared is not None else None,
            )
        self.epochDone = False

    # return state/action dict based upon passed state
    def act(self):
        """
        Act upon the previous observation.
        """
        if self.observation is not None and 'text' in self.observation:
            t = Message({'text': 'Hello agent!'})
        return t

    def epoch_done(self):
        """
        Return whether the epoch is done.
        """
        return self.epochDone

    # Default unknown length
    def num_examples(self):
        """
        Return the number of examples (e.g. individual utterances) in the dataset.

        Default implementation returns `None`, indicating an unknown number.
        """
        return None

    def num_episodes(self):
        """
        Return the number of episodes (e.g. conversations) in the dataset.

        Default implementation returns `None`, indicating an unknown number.
        """
        return None

    def report(self):
        """
        Return metrics showing total examples and accuracy if available.
        """
        return self.metrics.report()

    def reset(self):
        """
        Reset the teacher.
        """
        super().reset()
        self.reset_metrics()
        self.epochDone = False

    def reset_metrics(self):
        """
        Reset metrics.
        """
        self.metrics.clear()

    def share(self):
        """
        In addition to default Agent shared parameters, share metrics.
        """
        shared = super().share()
        shared['metrics'] = self.metrics.share()
        return shared

    def __iter__(self):
        """
        Iterate through the examples of the teacher.
        """
        clone = self.clone()
        while True:
            message = clone.act()
            if not isinstance(message, Message):
                # backwards compatibility with older agents
                message = Message(message)
            if message.is_padding():
                break
            yield message


class FixedDialogTeacher(Teacher):
    """
    A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - Provides a threadpool option for loading data (especially useful for
      large data, e.g. images)

    In order to take advantage of the first few features, all a subclass has to
    implement is three functions: ``num_episodes``, ``num_examples``, and
    ``get`` (which returns a specific example from a specific episode).

    To utilize the DataLoader for threadpool loading, a teacher should
    implement the ``submit_load_request`` function to send a load request
    to the DataLoader by calling ``self.data_loader.request_load`` with the
    appropriate arguments (``receive_fn, load_fn, args``). The DataLoader then
    returns the data to the teacher's ``data_queue``, which the teacher can
    poll in its ``act`` method.

    The following is an example of the DataLoader usage in the VQA-V1 teacher.

    1. In the teacher's ``init`` function, the teacher calls its
       ``submit_load_request`` function to preload an image.
    2. The ``submit_load_request`` function gets the next ``episode_idx``,
       and computes the image path for the load request.
    3. At the end of ``submit_load_request``, the teacher calls
       ``self.data_loader.request_load`` with three args:

        - ``self.receive_data`` - the function that the DataLoader calls to
          return the the loaded object
        - ``self.image_loader.load`` - the function used to load the image
          from the image path
        - ``[img_path]`` - a list of arguments for the load function, which
          in this case is the path of the image.

    4. In the teacher's ``act`` function, the teacher loads the data from
       its data queue.
    5. At the end of the ``act`` function, the teacher calls
       ``submit_load_request`` to preload an image for the next example.

    To see this in action, take a look at this teacher in ``tasks.vqa_v1.agents``.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'random'):
            self.random = self.datatype == 'train'
        if not hasattr(self, 'training'):
            self.training = DatatypeHelper.is_training(self.datatype)
        if not hasattr(self, 'cycle'):
            self.cycle = DatatypeHelper.should_cycle(self.datatype)
        if not hasattr(self, 'datafile'):
            self.datafile = opt.get('datafile')
        # set up support for multithreaded data loading
        self.data_queue = queue.Queue()
        if shared:
            self.index = shared['index']
            if 'data_loader' in shared:
                self.data_loader = shared['data_loader']
            if 'threadindex' in shared:
                self.threadindex = shared['threadindex']
            if 'examples' in shared:
                self.examples = shared['examples']
        else:
            self.index = AttrDict(value=-1)

        if not hasattr(self, 'data_loader'):
            if opt.get('background_index') is None:
                self.data_loader = DataLoader(opt)
            else:
                self.data_loader = _ErrorThrowingDataLoader(opt)
            self.data_loader.start()

        # set up batching
        self.bsz = opt.get('batchsize', 1)

        if shared:
            self.mutators = shared.get('mutators', [])
        else:
            mutator_types = Mutator.load_mutator_types(self.opt.get('mutators'))
            self.mutators = [mutator(self.opt) for mutator in mutator_types]

        self._episode_done = True

    def reset(self):
        """
        Reset the dialog to the start of the epoch, and reset all metrics.
        """
        super().reset()
        self.metrics.clear()
        self.lastY = None
        self.last_act = None
        self._episode_done = True
        self.epochDone = False
        self.data_queue = queue.Queue()

        self.episode_idx = -1
        self.index.value = -1

    def submit_load_request(self):
        """
        Submit a load request.

        An agent should implement this method to submit requests to the data
        loader. At the end of this method, the agent should call
        ``self.data_loader.request_load()`` with the appropriate args.

        By default, this method does nothing.
        """
        # TODO: mark as abstract
        pass

    def receive_data(self, future: concurrent.futures.Future):
        """
        Receive data from the data loader.

        :param future: result from the load request.
        """
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        """
        Share the data and dataloader.
        """
        shared = super().share()

        if hasattr(self, 'examples'):
            shared['examples'] = self.examples

        if hasattr(self, 'data_loader'):
            shared['data_loader'] = self.data_loader

        if hasattr(self, 'mutators'):
            shared['mutators'] = self.mutators

        shared['index'] = self.index

        return shared

    def next_episode_idx(self, num_eps=None, loop=None):
        """
        Return the next episode index.

        :param num_eps:
            default None uses ``num_episodes`` value.
        :param loop:
            default None loops during training but not evaluation.
        """
        if num_eps is None:
            num_eps = self.num_episodes()
        if loop is None:
            loop = self.training
        if self.random:
            new_idx = random.randrange(num_eps)
        else:
            self.index.value += 1
            if loop:
                try:
                    self.index.value %= num_eps
                except ZeroDivisionError:
                    raise ZeroDivisionError(
                        "The teacher has either empty data (e.g. setup_data yielded "
                        "no items, or self.num_episodes() == 0). We do not support "
                        "empty datasets (or folds) at this time."
                    )
            new_idx = self.index.value
        return new_idx

    def next_example(self):
        """
        Return the next example.

        If there are multiple examples in the same episode, returns the next one in that
        episode. If that episode is over, gets a new episode index and returns the first
        example of that episode.
        """
        if self._episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0

            if self.episode_idx >= self.num_episodes():
                return Message.padding_example(), True

            # buffer the full conversation ahead of time for mutators
            episode_buffer = []
            buffer_entry_idx = 0
            while True:
                entry = self.get(self.episode_idx, buffer_entry_idx)
                if not isinstance(entry, Message):
                    assert isinstance(entry, dict)
                    typ = type(self)
                    warn_once(
                        f"{typ.__module__}.{typ.__name__}' is outputting dicts "
                        "instead of messages. If this is a teacher that is part of "
                        "ParlAI, please file an issue on GitHub. If it is your own "
                        "teacher, please return a Message object instead."
                    )
                    entry = Message(entry)
                episode_buffer.append(entry)
                if entry.get('episode_done'):
                    break
                buffer_entry_idx += 1
            # apply mutators
            if self.mutators:
                episode_buffer = [m.copy() for m in episode_buffer]
                for mutator in self.mutators:
                    episode_buffer = mutator(episode_buffer)
            self.episode_buffer = list(episode_buffer)

            if not self.episode_buffer:
                # if we got back an empty episode after mutating, skip it
                return self.next_example()
        else:
            self.entry_idx += 1

        if self.episode_idx >= self.num_episodes():
            return Message.padding_example(), True

        # buffer the entire conversation so we can apply mutators
        ex = self.episode_buffer[self.entry_idx]
        self._episode_done = self.entry_idx == len(self.episode_buffer) - 1

        if (
            not self.cycle
            and self._episode_done
            and self.episode_idx + self.opt.get("batchsize", 1) >= self.num_episodes()
        ):
            epoch_done = True
        else:
            epoch_done = False

        return ex, epoch_done

    def num_episodes(self) -> int:
        """
        Get the number of episodes in this dataset.
        """
        raise RuntimeError('"num_episodes" must be overridden by children.')

    def num_examples(self) -> int:
        """
        Get the total number of examples in this dataset.
        """
        raise RuntimeError('"num_examples" must be overridden by children.')

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        Children must override this method in order to inherit the
        `next_example` method.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode.  Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        # TODO: mark as abstract, get rid of runtime error.
        raise RuntimeError('"Get" method must be overridden by children.')

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        self.metrics.clear_recent()
        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.evaluate_response(observation, self.lastY)
            self.custom_evaluation(self.last_act, self.lastY, observation)
            self.lastY = None
        recent_metrics = self.metrics.report_recent()
        if recent_metrics:
            # for display purposes (display_model), take all accumulated
            # metrics back into the original observation. This is an abuse of
            # Messages being pointers
            if 'metrics' in observation:
                # override agent-level metrics if present
                observation.pop('metrics')
            observation['metrics'] = recent_metrics
        return observation

    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        """
        A method designated for hooking custom evaluations into teachers.

        Generally, a user will want to use `self.metrics.add` to record any
        specialized metrics that only make sense for this one dataset.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels, if there were any.
        :param model_response:
            The raw response from the model. Generally you want to rely on the
            text field, but others may be necessary in specific situations.
        """
        pass

    def act(self):
        """
        Send new dialog message.
        """
        orig_action = self.get_orig_action()
        processed_action = self.process_action(orig_action)
        return processed_action

    def get_orig_action(self) -> Message:
        """
        Get the unprocessed action and reset if needed.

        This function will return the raw action from `self.next_example()`, before the
        `self.last_act` and `self.lastY` attributes have been defined based on this
        action for metrics or custom evaluations. This is so that wrapper teachers can
        modify the raw action first, such as to change the contents of its 'text' and
        'label' fields, without the action becoming out of sync with `self.last_act` and
        `self.lastY`.
        """
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next example, action is episode_done dict if already out of exs
        action, self.epochDone = self.next_example()
        if not isinstance(action, Message):
            # TODO: all teachers should eventually create messages
            # while setting up the data, so this won't be necessary
            action = Message(action)

        return action

    def process_action(self, action: Message) -> Message:
        """
        Remember the raw action and prepare its fields for passing out of the teacher.
        """
        action.force_set('id', self.getID())

        # remember correct answer if available
        self.last_act = action
        self.lastY = action.get('labels', action.get('eval_labels', None))
        if not DatatypeHelper.is_training(self.datatype) and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action = action.copy()
            labels = action.pop('labels')
            if not self.opt.get('hide_labels', False):
                action['eval_labels'] = labels

        return action


class DialogTeacher(FixedDialogTeacher):
    """
    A base teacher class for doing dialog with fixed chat logs.

    This class provides a set a basic functionality:

    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data

    In order to subclass this class, you must implement ``setup_data()`` in
    your class, which reads your data file as an iterator.
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        if not hasattr(self, 'setup_data'):
            raise RuntimeError(
                'Must implement setup_data or subclass a class '
                'which implements it (e.g. FbDeprecatedDialogTeacher) '
                'in order to use this class.'
            )
        super().__init__(opt, shared)

        self.datatype = opt['datatype']
        self.training = DatatypeHelper.is_training(self.datatype)
        self.cycle = DatatypeHelper.should_cycle(self.datatype)
        self.stream = 'stream' in self.datatype

        # first initialize any shared objects
        data_class = StreamDialogData if self.stream else DialogData
        kwargs = (
            # never cycle if "ordered" is in the datatype. this is used by
            # build_dict to enumerate through the data exactly once while still
            # marking examples as training examples.
            {'cycle': self.cycle}
            if self.stream
            else {}
        )
        if shared and shared.get('data'):
            self.data = data_class(opt, shared=shared['data'], **kwargs)
        else:
            if 'datafile' not in self.opt:
                raise KeyError(
                    ERROR_MESSAGE_NO_DATAFILE.format(class_name=self.__class__.__name__)
                )
            self.data = data_class(
                opt,
                data_loader=self.setup_data,
                cands=self.label_candidates(),
                **kwargs,
            )

        self.reset()

    @abstractmethod
    def setup_data(self, datafile: str):
        """
        The core method which the user should override.

        Yields the data, one message at a time, as well as markers indicating
        new episodes.

        :param str datafile:
            If the initializer set a 'datafile' field within the initialization,
            this will be provided here. Otherwise, datafile will be the fold:
            either "train", "valid", or "test".

        :return:
            Yields pairs (message, new_episode) containing a Message object
            and whether the message marks the beginning of a totally new
            episode.
        """
        pass

    def reset(self):
        """
        Reset the dialog to the start of the epoch, reset all metrics.
        """
        super().reset()
        if self.stream:
            self.data.reset()
            self.epochDone = False

    def share(self):
        """
        Share the data.
        """
        shared = super().share()
        if hasattr(self, 'data'):
            shared['data'] = self.data.share()
        return shared

    def label_candidates(self):
        """
        Provide consistent label candidates for all examples.

        Default implementation returns ``None`` always, but this may be overridden to
        provide candidates in all areas. See ``FbDialogueTeacher``.
        """
        # TODO DEPRECATIONDAY: FbDialogueTeacher is being deprecated, should we
        # remove this?

        # TODO: mark as optionally abstract?
        return None

    def num_episodes(self) -> int:
        """
        Return the number of episodes in the data.
        """
        if hasattr(self, "_num_episodes_cache"):
            return self._num_episodes_cache
        try:
            return self.data.num_episodes()
        except AttributeError:
            return super().num_episodes()

    def num_examples(self) -> int:
        """
        Return the number of examples in the data.
        """
        if hasattr(self, '_num_examples_cache'):
            return self._num_examples_cache
        try:
            self._num_examples_cache: int = self.data.num_examples()
        except AttributeError:
            self._num_examples_cache = super().num_examples()
        return self._num_examples_cache

    def get(self, episode_idx, entry_idx=0):
        """
        Get a specific example.
        """
        return self.data.get(episode_idx, entry_idx)[0]

    def next_example(self):
        """
        Get the next example.
        """
        if self.stream:
            # unfortunately we need to also do the mutator buffering here.
            # it's difficult to structure it so it's not
            if hasattr(self, 'episode_buffer') and self.episode_buffer:
                action = self.episode_buffer.pop(0)
                epoch_done = (not self.episode_buffer) and self._saw_epoch_done
                return action, epoch_done
            episode_buffer = []
            while True:
                action, epoch_done = self.data.get()
                episode_buffer.append(action)
                if action['episode_done']:
                    self._saw_epoch_done = epoch_done
                    break
            # perform any mutations there are
            if self.mutators:
                episode_buffer = [m.copy() for m in episode_buffer]
                for mutator in self.mutators:
                    episode_buffer = mutator(episode_buffer)
            # make sure mutations are fully realized (not generators)
            self.episode_buffer = list(episode_buffer)
            # The recursive call has dual purpose:
            # - if we get back an empty episode after mutating, skip it gracefully
            # - pull the first item the episode w/ epoch_done logic, but DRY
            return self.next_example()
        else:
            action, epoch_done = super().next_example()
        return action, epoch_done


class DialogData(object):
    """
    Provides a data structure for accessing textual dialog data.

    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    ``DialogTeacher`` class.

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).

    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.

    The contents of the ``((x, y, r, c, i), new_episode?)`` tuples returned by
    the data loader is the following:

    - ``x`` (str) is a query and possibly context
    - ``y`` (iter) is an iterable of label(s) for that query
    - ``r`` (str) is the str reward for getting that query correct
    - ``c`` (iter) is an iterable of label candidates that the student can choose from
    - ``i`` (str) is a str path to an image on disk, which will be loaded by the
      data class at request-time. should always point to the raw image file.
    - ``new_episode?`` (bool) is a boolean value specifying whether that example
      is the start of a new episode. If you don't use episodes set this
      to ``True`` every time.
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # in case we need to shard the dataset
        self.rank = get_rank()
        self.num_workers = num_workers()
        self.is_distributed_and_is_eval = is_distributed() and any(
            x in opt['datatype'] for x in ('valid', 'test', 'train:evalmode')
        )

        # self.data is a list of episodes
        # each episode is a tuple of entries
        # each entry is a tuple of values for the action/observation table
        if shared:
            self.image_loader = shared.get('image_loader', None)
            self.data = shared.get('data', [])
            self.cands = shared.get('cands', None)
        else:
            self.image_loader = ImageLoader(opt)
            self.data = []

            if 'datafile' not in opt:
                raise KeyError(
                    ERROR_MESSAGE_NO_DATAFILE.format(class_name=self.__class__.__name__)
                )

            self._load(data_loader, opt['datafile'])
            self.cands = None if cands is None else set(c for c in cands)

        self.addedCands = []
        self.copied_cands = False

    def share(self):
        """
        Share the data.
        """
        shared = {
            'data': self.data,
            'cands': self.cands,
            'image_loader': self.image_loader,
        }
        return shared

    def _read_episode(self, data_loader):
        """
        Read one episode at a time from the provided iterable over entries.

        :param data_loader:
            an iterable which returns tuples in the format described in the
            class docstring.
        """
        episode = []
        for entry, new in data_loader:
            if new and len(episode) > 0:
                yield episode
                episode = []

            episode.append(entry)

        if len(episode) > 0:
            yield episode

    def _load(self, data_loader, datafile):
        """
        Load up data from an iterable over tuples described in the class docs.

        :param iter data_loader:
            an iterator which returns tuples in the format described in the
            class docstring.
        :param str datafile:
        """
        for i, episode in enumerate(self._read_episode(data_loader(datafile))):
            if not self.is_distributed_and_is_eval or i % self.num_workers == self.rank:
                self.data.append(episode)

    def num_episodes(self):
        """
        Return number of episodes in the dataset.
        """
        return len(self.data)

    def num_examples(self):
        """
        Return total number of entries available.

        Each episode has at least one entry, but might have many more.
        """
        if hasattr(self, '_num_examples_cache'):
            return self._num_examples_cache
        self._num_examples_cache = sum(len(episode) for episode in self.data)
        return self._num_examples_cache

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode. Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        if episode_idx >= len(self.data):
            return Message.padding_example(), True
        next_episode_idx_for_rank = episode_idx + 1
        # first look up data
        episode = self.data[episode_idx]
        entry = episode[entry_idx]
        episode_done = entry_idx == len(episode) - 1

        end_of_data = episode_done and next_episode_idx_for_rank >= len(self.data)

        # now pack it in a action-observation dictionary
        table = self.build_table(entry)

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

    def build_table(self, entry):
        """
        Packs an entry into an action-observation dictionary.

        :param entry: a tuple in the form described in the class docstring.
        """
        if isinstance(entry, (dict, Message)):
            # user is already provided things
            if 'eval_labels' in entry or 'eval_label' in entry:
                raise KeyError(
                    'Labels are converted to eval_labels automatically. Please do not '
                    'set them in setup_data.'
                )
            if 'episode_done' in entry:
                raise KeyError(
                    "episode_done is set automatically for you. Please don't set it "
                    "in setup_data."
                )
            if 'label' in entry:
                # for convenience, rename to the labels convention automatically
                label = entry.pop('label')
                assert isinstance(label, str)
                entry['labels'] = (label,)
            if 'labels' in entry and isinstance(entry['labels'], str):
                entry['labels'] = (entry['labels'],)
            table = entry.copy()
        elif isinstance(entry, (Tuple, List)):
            table = {}
            if entry[0] is not None:
                table['text'] = entry[0]
            if len(entry) > 1 and entry[1] is not None:
                l = entry[1]
                if isinstance(l, str):
                    l = (l,)
                table['labels'] = l
            if len(entry) > 2 and entry[2] is not None:
                table['reward'] = entry[2]
            if len(entry) > 3 and entry[3] is not None:
                table['label_candidates'] = entry[3]
            if len(entry) > 4 and entry[4] is not None:
                img = self.image_loader.load(entry[4])
                if img is not None:
                    table['image'] = img
        else:
            raise TypeError(
                f"items out of setup_data should be dict, Message, list, or tuple. "
                f"Got {type(entry)})"
            )

        if table.get('labels', None) is not None and self.cands is not None:
            if self.addedCands:
                # remove elements in addedCands
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    # add labels, queue them for removal next time
                    if not self.copied_cands:
                        self.cands = self.cands.copy()
                        self.copied_cands = True
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands

        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')

        # go ahead and make it a message
        if isinstance(table, dict):
            table = Message(table)

        return table


class StreamDialogData(DialogData):
    """
    Provides a data structure for streaming textual dialog data.

    This can be used whenever the dialog data follows the format described in
    DialogData but cannot fit entirely into memory.

    Additional keyword-argument cycle defines if the stream should restart from
    the beginning after an epoch is finished (defaults to True).

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).
    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.
    :param cycle:
        (default True) whether to restart at beginning when end of stream
        reached without reset being called.
    """

    # represents that we haven't read in any data at all
    _FIRST_PASS = None
    # represents that we are out of data.
    _END_OF_EPOCH = -1

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # super() call initiates stream in self.data by calling _load()
        super().__init__(opt, data_loader, cands, shared, **kwargs)
        self.cycle = kwargs['cycle'] if 'cycle' in kwargs else True

        if shared:
            # auxiliary instances hold pointer to main datastream in self.data
            self.reset_data = shared['reset']
            # Share datafile and data_loader for computing num_exs and num_eps
            self.datafile = shared['datafile']
            self.length_datafile = opt.get('length_datafile', None)
            self.data_loader = shared['data_loader']
            if 'lock' in shared:
                self.lock = shared['lock']
        else:
            # main instance holds the stream and shares pointer to it
            self.data_loader = data_loader
            if 'datafile' not in opt:
                raise KeyError(
                    ERROR_MESSAGE_NO_DATAFILE.format(class_name=self.__class__.__name__)
                )
            self.datafile = opt['datafile']
            self.length_datafile = opt.get('length_datafile', None)
            self.reset_data = None
            self.is_reset = True
        self.entry_idx = 0
        self.cur_episode = self._FIRST_PASS
        self.num_eps = None
        self.num_exs = None

        self.rank = get_rank()
        self.num_workers = num_workers()
        self.is_distributed_and_is_eval = (
            self.num_workers > 1 and not DatatypeHelper.is_training(opt['datatype'])
        )

    def share(self):
        """
        Share the stream.
        """
        shared = super().share()
        # also share reset method to allow datastream to be reset
        shared['reset'] = self.reset
        # share datafile and data for loading length if necessary
        shared['datafile'] = self.datafile
        shared['data_loader'] = self.data_loader
        if hasattr(self, 'lock'):
            shared['lock'] = self.lock

        return shared

    def _load(self, data_loader, datafile):
        """
        Load data generator into data field.
        """
        self.data = self._data_generator(data_loader, datafile)

    def _data_generator(self, data_loader, datafile):
        """
        Generate data using the iterator over tuples constructed by data_loader.
        """
        self.is_reset = False
        idx = 0
        while True:
            for episode in self._read_episode(data_loader(datafile)):
                # We only shard the data set at evaluation time, as training is
                # done using sampling-with-replacement.
                if not self.is_distributed_and_is_eval or (
                    idx % self.num_workers == self.rank
                ):
                    yield episode
                idx += 1
            while not self.cycle:
                yield self._END_OF_EPOCH

    def load_length(self):
        """
        Calculate the length of the dataset and caches it in a file.

        Note that this can take some time for large datasets. Episode and entry indexes
        cannot be specified during streaming.
        """
        if self.length_datafile:
            length_file = self.length_datafile
        else:
            datafiles = (
                self.datafile if type(self.datafile) is tuple else [self.datafile]
            )
            length_file = datafiles[0] + ".lengths"
        if not PathManager.exists(length_file):
            num_eps = 0
            num_exs = 0
            for episode in self._read_episode(self.data_loader(self.datafile)):
                num_eps += 1
                num_exs += len(episode)
            with PathManager.open(length_file, 'w', encoding="utf-8") as f:
                f.write("{}\n{}".format(num_eps, num_exs))
        else:
            with PathManager.open(length_file, 'r', encoding='utf-8') as f:
                num_eps, num_exs = f.readlines()
        return int(num_eps), int(num_exs)

    def num_examples(self):
        """
        Return the number of examples in the data.
        """
        if not self.num_exs:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes in the data.
        """
        if not self.num_eps:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_eps

    def get(self):
        """
        Get the next entry from the stream.

        When episode is done returns first entry of next episode.
        """
        if self.cur_episode is self._FIRST_PASS:
            # first go around, always read off the episode
            # maybe lock this line
            self.cur_episode = next(self.data)
        if self.cur_episode == self._END_OF_EPOCH:
            # we're done here
            return Message.padding_example(), True
        entry = self.cur_episode[self.entry_idx]
        table = self.build_table(entry)
        episode_done = self.entry_idx == len(self.cur_episode) - 1
        table['episode_done'] = episode_done
        if episode_done:
            # maybe lock this line
            self.cur_episode = next(self.data)
            self.entry_idx = 0
        else:
            self.entry_idx += 1
        return table, self.cur_episode == self._END_OF_EPOCH

    def reset(self):
        """
        Reset the datastream to its beginning.
        """
        if self.reset_data is not None:
            # auxiliary instance, reset main datastream
            self.data = self.reset_data()
        elif not self.is_reset:
            # if main instance is not reset, reset datastream
            self._load(self.data_loader, self.datafile)
            self.is_reset = True
        self.entry_idx = 0
        self.cur_episode = self._FIRST_PASS
        return self.data


class FbDeprecatedDialogTeacher(DialogTeacher):
    """
    This module provides access to data in the Facebook Dialog format.

    Subclasses ``DialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "fbdialog" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way FB Dialog data is set up is as follows:

    ::

        1 Sam went to the kitchen.
        2 Pat gave Sam the milk.
        3 Where is the milk?<TAB>kitchen<TAB>1<TAB>hallway|kitchen|bathroom
        4 Sam went to the hallway.
        5 Pat went to the bathroom.
        6 Where is the milk?<TAB>hallway<TAB>1<TAB>hallway|kitchen|bathroom

    Lines 1-6 represent a single episode, with two different examples: the
    first example is lines 1-3, and the second is lines 4-6.

    Lines 1,2,4, and 5 represent contextual information.

    Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog in this format can contain any speech, not just QA pairs:

    ::

        1 Hi how's it going?<TAB>It's going great. What's new?
        2 Well I'm working on a new project at work.<TAB>Oh me too!
        3 Oh cool!<TAB>Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.cloze = opt.get('cloze', False)
        if shared and 'cands' in shared:
            self.cands = shared['cands']
        else:
            self.cands = self.load_cands(opt.get('cands_datafile', None))
        super().__init__(opt, shared)

    def share(self):
        """
        Share the data and candidates.
        """
        shared = super().share()
        shared['cands'] = self.cands
        return shared

    def label_candidates(self):
        """
        Return the candidates.
        """
        return self.cands

    def load_cands(self, path):
        """
        Load a global fixed set of candidates.

        The candidates will be provided by the teacher for every example (the true
        labels for a specific example are also added to this set, so that it's possible
        to get the right answer).
        """
        if path is None:
            return None
        cands = []
        lines_have_ids = False
        cands_are_replies = False
        cnt = 0
        with PathManager.open(path, encoding='utf-8') as read:
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) > 0:
                    cnt = cnt + 1
                    # If lines are numbered we strip them of numbers.
                    if cnt == 1 and line[0:2] == '1 ':
                        lines_have_ids = True
                    # If tabs then the label_candidates are all the replies.
                    if '\t' in line and not cands_are_replies:
                        cands_are_replies = True
                        cands = []
                    if lines_have_ids:
                        space_idx = line.find(' ')
                        line = line[space_idx + 1 :]
                        if cands_are_replies:
                            sp = line.split('\t')
                            if len(sp) > 1 and sp[1] != '':
                                cands.append(sp[1])
                        else:
                            cands.append(line)
                    else:
                        cands.append(line)
        return cands

    def setup_data(self, path):
        r"""
        Read data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.

        ``x`` represents a query, ``y`` represents the labels, ``r`` represents
        any reward, and ``c`` represents any label_candidates.

        The example above will be translated into the following tuples:

        ::

            x: 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)


        ::

            x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        logging.info(f"loading fbdialog data: {path}")
        with PathManager.open(path, encoding='utf-8') as read:
            start = True
            x = ''
            reward = 0
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    # empty response
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                if space_idx == -1:
                    # empty line, both individuals are saying whitespace
                    conv_id = int(line)
                else:
                    conv_id = int(line[:space_idx])

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>label_candidates
                # where y, reward, and label_candidates are optional
                split = line[space_idx + 1 :].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word
                # Empty reward string same as None
                if len(split) > 2 and split[2] == '':
                    split[2] = None

                # now check if we're at a new episode
                if last_conv_id is None or conv_id <= last_conv_id:
                    x = x.strip()
                    if x:
                        yield [x, None, reward], start
                    start = True
                    reward = 0
                    # start a new episode
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(
                            x=split[0]
                        )
                    else:
                        x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        x = split[0]
                last_conv_id = conv_id
                if len(split) > 2 and split[2]:
                    reward += float(split[2])

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split label_candidates
                        split[3] = split[3].split('|')
                    if len(split) > 2:
                        split[2] = reward
                    else:
                        split.append(reward)
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = 0
            if x:
                yield [x, None, reward], start


class ParlAIDialogTeacher(FixedDialogTeacher):
    """
    This module provides access to data in the ParlAI Text Dialog format.

    Subclasses ``FixedDialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "ParlAI text" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way the data is set up is as follows:

    ::

        text:Sam went to the kitchen. <NEWL>
        Pat gave Sam the milk. <NEWL>
        Where is the milk? <TAB> labels:kitchen <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom
        text:Sam went to the hallway. <NEWL>
        Pat went to the bathroom. <NEWL>
        Where is the milk? <TAB> labels:hallway <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom <TAB> episode_done:True

    Lines 1-2 represent a single episode, with a different example on each line.
    The lines contain a query and a label for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog this format can contain any speech, not just QA pairs:

    ::

        text:Hi how's it going?<TAB>labels:It's going great. What's new?
        text:Well I'm working on a new project at work.<TAB>labels:Oh me too!
        text:Oh cool!<TAB>labels:Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.episodes = []
            self.num_exs = 0
            if opt.get('parlaidialogteacher_datafile') is not None:
                self._setup_data(opt.get('parlaidialogteacher_datafile'))
        else:
            self.episodes = shared['episodes']
            self.num_exs = sum(len(e) for e in self.episodes)

        self.id = opt['task']

        self.reset()

    def share(self):
        """
        Share the episodes.
        """
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_examples(self):
        """
        Return the number of examples from the data.
        """
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes from the data.
        """
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        return self.episodes[episode_idx][entry_idx]

    def _setup_data(self, path):
        logging.info(f"Loading ParlAI text data: {path}")
        self.episodes = []
        self.num_exs = 0
        eps = []
        with PathManager.open(path, newline='\n', encoding='utf-8') as read:
            for line_no, line in enumerate(read, 1):
                msg = str_to_msg(line.rstrip('\n'))
                if msg and 'eval_labels' in msg:
                    raise ValueError(
                        f"It looks like you've written eval_labels as a key in your "
                        f"data file. This is not appropriate; labels will be converted "
                        f"for you automatically. This is happening on Line {line_no} "
                        f"in {path}. The line is:\n\t{line}"
                    )
                if msg and 'text' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "text" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg and 'labels' not in msg:
                    raise ValueError(
                        f'ParlaiDialogTeacher requires a "labels" field in every '
                        f'entry, but one is missing in Line {line_no} in {path}. '
                        f'The line is:\n\t{line}'
                    )
                if msg:
                    self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            self.episodes.append(eps)
        if len(self.episodes) == 1 and line_no > 100:
            logging.error(
                f'The data in {path} looks like one very long episode. If this '
                f'is intentional, you may ignore this, but you MAY have a bug in '
                f'your data.'
            )


class YamlTeacher(DialogTeacher):
    """
    Teacher which loads data generated by `parlai.utils.testing.AutoTeacherTest`.
    """

    def __init__(self, opt, shared=None):
        # TODO: if we get rid of the streaming datafile num_episodes/num_examples
        # cache then we can support streaming here. but for now let's
        # just hardcode it
        opt = opt.copy()
        opt['datatype'] = opt['datatype'].replace(':stream', '')
        super().__init__(opt, shared=shared)

    def setup_data(self, datafile):
        with PathManager.open(datafile) as f:
            records = yaml.safe_load(f)
            next_episode_new = True
            for act in records['acts']:
                act = act[0]  # yaml wraps in a weird singleton list
                next_episode_new = act.pop('episode_done')
                if 'eval_labels' in act:
                    act['labels'] = act.pop('eval_labels')
                yield act, next_episode_new


class ConversationTeacher(DialogTeacher):
    """
    This module provides access to data in the Conversations format.

    Subclasses ``DialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "Conversations" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The data should be set up so that each dialogue instance (or, episode)
    occupies one line of valid JSON. The way the data is set up is as follows:

    ::
    { "dialog": [ [ {"id": "partner1", "text": "hello!"},  {"id": "partner2", "text": "hi back!"}  ] ] }

    NOTE: If the data is not on one line per dialogue, it will not load.
    Further, note that by default, dialogs are interpreted as being one-way.
    For example, consider this dialog (not that the data below is not on:

    ::

        {
            "dialog":[ [
                {"id":"modelx", "text": X1},
                {"id":"modely", "text": Y1},
                {"id":"modelx", "text": X2},
                {"id":"modely", "text": Y2},
                {"id":"modelx", "text": X3},
                {"id":"modely", "text": Y3},
            ] ]
        }

    (Note: we use line breaks for readability above, but this data will not load as
    stated, it must be on one line.)

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated,
    forming one episode. However, Y1 => X2 and Y2 => X3 are not created as
    separate examples by default.
    To change this behavior, you can set ``opt['label_turns']`` or ``--label-turns flag``.
    The default value is 'secondspeaker' (i.e., the second speaker's utterances are
    used as labels), but 'firstspeaker' and 'both' are also options. In the
    case of 'both', two episodes are generated for each conversation.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = super().add_cmdline_args(parser, partial_opt)
        agent.add_argument(
            '--label-turns',
            type=str,
            help='which speaker to use as label',
            choices=['firstspeaker', 'secondspeaker', 'both'],
            default='secondspeaker',
        )
        return parser

    def __init__(self, opt, shared=None):
        if not opt.get('conversationteacher_datafile'):
            raise RuntimeError('conversationteacher_datafile not specified')

        opt = copy.deepcopy(opt)
        opt['datafile'] = opt.get('conversationteacher_datafile')
        self.label_turns = opt.get('label_turns')
        super().__init__(opt, shared)
        self.id = opt['task']

    def _return_episode_examples(self, episode):
        for idx, example in enumerate(episode):
            episode_begin = idx == 0
            if 'episode_done' in example:
                example.pop('episode_done')
            yield example, episode_begin

    def setup_data(self, path):
        logging.info(f"[loading data from json file into task: {path} ]")
        conversations = Conversations(path)
        for conv in conversations:
            if conv.context:
                warn_once(
                    'At least one of these conversations contains a context, which is not being used'
                )
            turns = [t for t in conv.turns if t.get('id') != 'context']
            if len(turns) != len(conv.turns):
                warn_once(
                    'At least one of these conversations contains a context within the dialogue, which is being discarded'
                )
            turns.insert(0, Message({'text': '__SILENCE__'}))
            # train on odd turns as labels (turns w/ first speaker)
            if self.label_turns in ['firstspeaker', 'both']:
                eps = self._get_ep_from_turns(turns[::2], turns[1::2])
                if eps:
                    for example, example_begins in self._return_episode_examples(eps):
                        yield example, example_begins

            # train on even turns as labels (turns w/ second speaker)
            if self.label_turns in ['secondspeaker', 'both']:
                eps = self._get_ep_from_turns(turns[1::2], turns[2::2])
                if eps:
                    for example, example_begins in self._return_episode_examples(eps):
                        yield example, example_begins

    def _get_ep_from_turns(self, xturns, yturns):
        eps = []
        for xturn, yturn in zip(xturns, yturns):
            turn = {}
            turn['text'] = xturn.get('text').strip()
            turn['labels'] = [yturn.get('text').strip()]
            eps.append(turn)
        return eps


class AbstractImageTeacher(FixedDialogTeacher):
    """
    Abstract class to allow easier creation of image + dialogue tasks.

    This class handles creating image features via ImageLoader if applicable
    (resnet, resnext variants) or loading existing image features from a dict
    path as per get_image_features_path().

    Important methods and properties (override in subclass if needed):

    - get_data_path(): where data file is found (default: <datapath>/<task>)
    - get_image_path(): where images found (default: <datapath>/<task>/images)
    - get_image_features_path(): dict of image features (default:
      <datapath>/<task>/image_features)
    - @property image_id_key: which key in data file objects represents image_id
    - @property text_key: which key in data file objects represents text

    Note: Assumes data files are named <dt>.json

    @abstractmethod image_id_to_image_path() must be implemented in subclass

    Example with the key defaults (but the keys can be customized):

    .. code-block:: python

        obs = {
            'text': <caption>,
            'image': <image features if specified else image>
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.task = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        self.data_path = self.get_data_path(opt)
        self.data = self.load_data(self.data_path, self.opt)
        self.datatype = DatatypeHelper.fold(opt['datatype'])

        # Example of available models: 'resnet152', 'resnext101_32x48d_wsl',
        # and ImageLoader supports other resnet and resnext models too
        # Raises an Exception if not valid
        self._validate_image_mode_name(opt.get('image_mode'))

        # IMPORTANT NOTE: this teacher will be instantiated twice. The first
        # by build_dict in which case the image_mode is to 'no_image_model' to
        # avoid calculating image features twice.
        self.image_mode = opt.get('image_mode')

        # Not using default image_mode parameter b/c there is a normalization
        # (or bug) somewhere in build_dict that is setting it to none
        self.include_image = opt.get('image_mode') != 'no_image_model'

        self.image_path = self.get_image_path(opt)
        self.image_loader = None
        self.image_features_dim = opt.get('image_features_dim')
        self.blank_image_features = torch.FloatTensor(self.image_features_dim).fill_(0)

        if shared and 'data' in shared:
            self.data = shared['data']
            self.image_loader = shared['image_loader']
            if 'image_features_dict' in shared:
                self.image_features_dict = shared['image_features_dict']
        elif self.include_image:
            self.setup_image_features(self.data_path)
        else:
            # This will happen when building the dictionary - is normal
            # build_dict sets image_mode to 'none'
            warn_once('AbstractImageTeacher self.include_image was False')
            self.image_features_dict = None

        # TODO: won't need this after we have proper logging levels set
        self.__verbose = False

        self.reset()

    def get_available_image_mode_names(self):
        """
        Available image model names.

        resnet and resnext variants available from the ImageLoader. resnext101_XXXXX_wsl
        is the open-sourced FB AI model (960m images, 1.5k hashtags, finetuned on
        ImageNet).
        """
        available_model_names = ImageLoader.get_available_model_names()
        return ['no_image_model', 'raw', 'ascii'] + available_model_names

    def _validate_image_mode_name(self, a):
        """
        Validate the image_mode passed in.

        Needed because image_mode used elsewhere in ParlAI is not always consistent with
        what the image teacher allows.
        """
        if not isinstance(a, str):
            raise argparse.ArgumentTypeError(
                '%s must be a string representing image model name' % a
            )
        available_model_names = self.get_available_image_mode_names()
        if a not in available_model_names:
            raise argparse.ArgumentTypeError(
                '\"%s\" unknown image model name. Choose from: %s. Currently suggested resnet is resnet152 and resnext is resnext101_32x48d_wsl.'
                % (a, available_model_names)
            )
        return a

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        # Be sure to call super() if overriding this method b/c
        # AbstractImageTeacher has necessary params
        parser = super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('AbstractImageTeacher Arguments')
        agent.add_argument(
            '--image-path',
            type=str,
            default=None,
            help='Optional argument to specify where images for dataset are'
            'stored if already downloaded. Most tasks will download the images'
            'if not present on the < datapath > / < task > _images / * and * if'
            'this argument is not specified.',
        )

        agent.add_argument(
            '--image-features-dim',
            type=int,
            default=2048,
            help='Specify the size of image features Tensors.',
        )
        return parser

    @property
    def image_id_key(self):
        """
        Which key in the input data dict objects uniquely identify each image.

        Common image keys are "image_id" or "image_num". May be implemented by subclass.
        """
        return 'image_id'

    @property
    def text_key(self):
        """
        Which key in the input data dict objects identifies the text.

        Common keys are "text" or "comment". May be implemented by subclass.
        """
        return 'text'

    @abstractmethod
    def image_id_to_image_path(self, image_id):
        """
        Get the path of the image on disk.

        Must be implemented by subclass.
        """
        pass

    def get_data_path(self, opt):
        """
        Determines path to the data file.
        """
        task_name = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        data_path = os.path.join(opt['datapath'], task_name)
        return data_path

    def get_image_path(self, opt):
        """
        Return the path to the data directory and to the image directory.

        Is based on opt fields: task, datatype (train, valid, test), datapath.

        Subclass can override this.
        """
        data_path = self.get_data_path(opt)
        if opt.get('image_path', None):
            image_path = opt['image_path']
        else:
            # other common choice: .join(opt['datapath'], task_name + '_images')
            image_path = os.path.join(data_path, 'images')

        return image_path

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Image features for the dataset images are stored here.

        Can be overridden in subclass to use custom paths. Image features can be
        manually copied into this directory or in the case of ImageLoader eligible
        models, they will be built and stored here if not already there.
        """
        # In default implementation, self.data_path already has task name added
        image_features_path = os.path.join(self.data_path, 'image_features')

        PathManager.mkdirs(image_features_path)

        return os.path.join(
            image_features_path, '%s_%s_%s_features_dict' % (task, image_model_name, dt)
        )

    def is_image_mode_buildable(self, model_name):
        """
        Is buildable if features can be calculated by ImageLoader.

        Users may wish to compute features for the dataset offline and use in the model,
        in which case, the image model should return False and get_image_features()
        should be overridden in subclass.
        """
        return model_name in ImageLoader.get_available_model_names()

    def load_data(self, data_path, opt):
        """
        Loading the data file, which is the index to the images and text.

        It is often a .json file with the name of the <datatype>.json (i.e.
        train.json). Stores in self.data.

        Can be override by subclass.
        """

        dt = DatatypeHelper.fold(opt['datatype'])

        # Sometimes file is named "val" instead of "valid"
        if dt not in ['train', 'valid', 'val', 'test']:
            raise Exception(
                'Unknown dt parameter: %s. Expected either "train", "valid", or "test".'
                % dt
            )

        # Assumes file is train.json or valid.json named
        data_file = os.path.join(self.data_path, '%s.json' % dt)

        # Load the text data and image number indexes
        with PathManager.open(data_file, encoding='utf-8') as f:
            self.data = json.load(f)

        if len(self.data) > 0 and self.image_id_key not in self.data[0]:
            # Data doesn't have a "image_id" like field so add the index in the file to the data
            for idx, d in enumerate(self.data):
                d[self.image_id_key] = idx

        return self.data

    def setup_image_features(self, data_path):
        """
        Load text and image data.

        The image features all live in dicts by default in <data_path>/
        image_features/ but get_image_features_path() above can be overridden by
        subclass to put them elsewhere.

        In the (very odd) case that the resnet or resnext dicts (models
        buildable using ImageLoader) are not found, we build them.
        """
        if self.image_mode in ['raw', 'ascii']:
            self.image_features_dict = None
            self.image_loader = ImageLoader(self.opt)
            return
        image_mode_features_dict_path = self.get_image_features_path(
            self.task, self.image_mode, self.datatype
        )

        if PathManager.exists(image_mode_features_dict_path):
            logging.info(
                f'Loading existing image features dict for model: {self.image_mode} at: {image_mode_features_dict_path}'
            )
            with PathManager.open(image_mode_features_dict_path, 'rb') as f:
                self.image_features_dict = torch.load(f, map_location='cpu')
        else:
            logging.warning('No existing image features, attempting to build.')
            if self.is_image_mode_buildable(self.image_mode):
                # TODO: Awkward to modify the input opt but needed to use
                # TODO: ImageLoader functionality. Is from comment_battle,
                # TODO: will refactor this at some point soon most likely
                image_loader_opt = self.opt.copy()
                image_loader_opt['image_mode'] = (
                    self.image_mode if self.include_image else 'no_image_model'
                )

                image_loader_opt['image_size'] = 256
                image_loader_opt['image_cropsize'] = 224
                self.image_loader = ImageLoader(image_loader_opt)

                # try to build with ImageLoader (i.e. resenet/resnext variants)
                self.image_features_dict = self._build_image_features_dict(
                    self.data_path, self.datatype, image_mode_features_dict_path
                )
            else:
                raise RuntimeError(
                    'Image model: %s is not buildable by ImageLoader but does'
                    'not already exist on disk as an image features dict for'
                    'this dataset.' % self.image_mode
                )

    def _build_image_features_dict(self, data_path, dt, store_dict_path):
        """
        Build resne(x)t image features with ImageLoader.

        (Or anything handleable by ImageLoader) and save to path. Only called if we
        haven't already built the dict before.
        """
        image_features_dict = {}
        total = len(self.data)
        import tqdm

        pbar = tqdm.tqdm(
            total=total,
            unit='cand',
            unit_scale=True,
            desc='Building image features dict for %s with ImageLoader.'
            % self.image_mode,
        )
        num = 0
        for ex in self.data:
            img_id = ex[self.image_id_key]
            img_path = self.image_id_to_image_path(img_id)
            image = self.image_loader.load(img_path).detach()
            # spatial features are [1, image_dim, spatial_dim, spatial_dim] tensors.
            # reduce non-spatial features to one-dimensional feature prior to saving.
            if not self.image_loader.is_spatial(self.image_mode):
                image = image[0, :, 0, 0]
            image_features_dict[img_id] = image
            num += 1
            pbar.update(1)
            if num % 1000 == 0:
                logging.debug(f'Processing image index: {num}')
        atomic_save(image_features_dict, store_dict_path)
        return image_features_dict

    def reset(self):
        super().reset()
        self.example = None

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.data)

    def get_image_features(self, example):
        """
        Get image features for example.

        Can be overridden in subclass for different behavior. For large datasets, it may
        be more appropriate to use the ImageLoader.load() method to load image features
        (as this is essentially streaming the features from disk, so that we do not have
        to load a large image feature dict in memory). #TODO Could be the default option
        if we are using -dt train:stream
        """
        if self.image_mode in ['raw', 'ascii']:
            try:
                image = self.image_loader.load(
                    self.image_id_to_image_path(example['image_id'])
                )
            except FileNotFoundError:
                # No Image Here
                image = None
            return image

        key = str(example[self.image_id_key])
        if not self.include_image or key not in self.image_features_dict:
            image_features = self.blank_image_features
        else:
            image_features = self.image_features_dict[key]
        return image_features

    def get(self, episode_idx, entry_idx=0):
        """
        Override this in subclass if your data should be handled in a different format.
        """
        example = self.data[episode_idx]
        image_features = self.get_image_features(example)
        return {
            'labels': [example[self.text_key]],
            'image': image_features,
            'episode_idx': episode_idx,
            'episode_done': True,
        }

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        if hasattr(self, 'image_features_dict'):
            shared['image_features_dict'] = self.image_features_dict
        return shared


class MultiTaskTeacher(Teacher):
    """
    MultiTaskTeacher which teaches multiple tasks.

    Creates a teacher that is actually a set of teachers each based on a task
    string -- each of these teachers will get called in turn,
    either randomly or in order.  They are all in the same world (they are the
    same agent switching tasks).

    The task string format is described for the ``create_task_agents()``
    function above.
    """

    def __init__(self, opt: Opt, shared=None):
        self.tasks: List[Agent] = []
        self.opt = opt

        self.id = opt['task']
        if shared and 'tasks' in shared:
            self.tasks = [create_agent_from_shared(t) for t in shared['tasks']]
        else:
            tasks = opt['task'].split(',')
            for k in tasks:
                k = k.strip()
                if k:
                    opt_singletask = copy.deepcopy(opt)
                    opt_singletask['task'] = k
                    self.tasks.extend(create_task_agent_from_taskname(opt_singletask))
        self.task_idx = -1
        self.new_task = True
        self.random = DatatypeHelper.should_shuffle(opt['datatype'])
        # Make multi-task task probabilities.
        self.cum_task_weights = [1] * len(self.tasks)
        self.task_choices = range(len(self.tasks))
        weights = self.opt.get('multitask_weights', [1])
        if weights == 'stochastic':
            weights = [t.num_episodes() for t in self.tasks]
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight

    def num_examples(self):
        """
        Return the number of examples.
        """
        if not hasattr(self, 'num_exs'):
            # num_examples is sum of all examples in all tasks
            tasks_num_exs = [t.num_examples() for t in self.tasks]
            if any(num is None for num in tasks_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(tasks_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            # num_episodes is sum of all num_episodes in all tasks
            tasks_num_eps = [t.num_episodes() for t in self.tasks]
            if any(num is None for num in tasks_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(tasks_num_eps)
        return self.num_eps

    def observe(self, observation):
        """
        Make an observation.
        """
        return self.tasks[self.task_idx].observe(observation)

    def act(self):
        """
        Act on the previous observation.
        """
        if self.new_task:
            self.new_task = False
            if self.random:
                # select random teacher
                self.task_idx = random.choices(
                    self.task_choices, cum_weights=self.cum_task_weights
                )[0]
            else:
                # do at most one full loop looking for unfinished task
                for _ in range(len(self.tasks)):
                    self.task_idx = (self.task_idx + 1) % len(self.tasks)
                    if not self.tasks[self.task_idx].epoch_done():
                        # if this task has examples ready, break
                        break
                if self.tasks[self.task_idx].epoch_done():
                    # all tasks are done, so return empty action table
                    return Message.padding_example()
        t = self.tasks[self.task_idx].act()
        if t['episode_done']:
            self.new_task = True
        return t

    def epoch_done(self):
        """
        Return whether all subtasks are completed.
        """
        for t in self.tasks:
            if not t.epoch_done():
                return False
        return True

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        """
        Report aggregated metrics across all subtasks.
        """
        return aggregate_named_reports(
            {t.getID(): t.report() for t in self.tasks},
            micro_average=self.opt.get('aggregate_micro', False),
        )

    def reset(self):
        """
        Reset all subtasks.
        """
        for t in self.tasks:
            t.reset()

    def reset_metrics(self):
        """
        Reset metrics for each subtask.
        """
        for t in self.tasks:
            t.reset_metrics()

    def share(self):
        """
        Shares this teacher by sharing each subtask.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['tasks'] = [t.share() for t in self.tasks]
        return shared

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for t in self.tasks:
            t.shutdown()


class ChunkTeacher(FixedDialogTeacher, ABC):
    """
    Useful for loading large amounts of data.

    Data is separated into chunks and loaded one chunk at a time. Loads the data off of
    the main thread.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.buffersize = self.get_buffersize()

        self.set_datasettings(opt)
        # chunk teacher makes shuffling decisions based on training, but
        # train:stream turns off shuffling in other teachers.
        self.datatype = DatatypeHelper.strip_stream(opt['datatype'])

        self.dws = int(self.opt.get('distributed_world_size', 1))
        self.rank = int(self.opt.get('rank', 0))
        self.bg_index = self.opt.get('background_index', None)

        # If we're in training mode with --num-workers > 0, we will run the
        # chunk teacher in single threaded mode (self.threading is False). In
        # this mode, we will block on chunk loading.

        # If we're not using --num-workers, or we're in validation/testing, we
        # _always_ run in normal threading mode, where chunk loading is pushed
        # to a background thread. However, since python threading is is blocked
        # by the GIL, this only manages to background I/O.

        # Potentially in the future, we may support --num-workers in validation,
        # in which case we can get rid of one of these.

        self.threading = not (opt.get('num_workers', 0) > 0 and self.is_train)
        if not self.threading and opt.get('background_index') is None:
            # don't start loading data on the main driver, we don't need it
            opt['no_auto_enqueues'] = True
        if not self.threading:
            # if we're in single-threaded (background preprocessing mode), we
            # can't have a max queue size or we will hang if we overfill it
            self.buffersize = 0

        if shared is not None:
            self.is_root_teacher = False
            self.chunks = shared['chunks']
            self.samples = shared['samples']
            self.reset_counter = shared['reset_counter']
            self.rng = shared['rng']
            self.tot_samples_loaded = shared['tot_samples_loaded']
        else:
            self.is_root_teacher = True
            self.samples = queue.Queue(maxsize=self.buffersize)
            self.chunks = queue.Queue()
            self.reset_counter = SimpleCounter()  # track no. of resets
            if DatatypeHelper.should_shuffle(self.datatype):
                # TODO: possible need a fixed seed here in the future
                self.rng = random.Random()
            else:
                self.rng = random.Random(42)
            self._enqueue_chunks()
            # launch queue loader on the main thread
            self.tot_samples_loaded = defaultdict(int)
            if not opt.get("no_auto_enqueues", False):
                # we only enqueue the train thread because the reset() called at
                # the top of training will handle this
                self._enqueue_request()

        self._episode_done = True
        self.last_queue_output = None

    def _get_data_folder(self):
        if not self.opt.get('datafile'):
            raise RuntimeError(
                'Must specify datafile or override this function (_get_data_folder) '
                'to return the data folder.'
            )

        return self.opt['datafile']

    @abstractmethod
    def get_num_samples(self, opt: Opt) -> Tuple[int, int]:
        """
        [Abstract] Return the number of samples.

        Returns a tuple of (num_examples, num_episodes) based on the data split.
        """
        pass

    @abstractmethod
    def get_fold_chunks(self, opt: Opt) -> List[int]:  # type: ignore
        """
        [Abstract] Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        pass

    def get_buffersize(self):
        """
        Size of buffer.

        Override this in your child class to change the buffer size.
        """
        return 100000

    def set_datasettings(self, opt: Opt):
        self.folder = self._get_data_folder()
        self.num_exs, self.num_eps = self.get_num_samples(opt)
        self.fold_chunks = self.get_fold_chunks(opt)

        self.is_train = DatatypeHelper.is_training(opt['datatype'])

    def share(self):
        shared = super().share()
        shared['samples'] = self.samples
        shared['chunks'] = self.chunks
        shared['reset_counter'] = self.reset_counter
        shared['rng'] = self.rng
        shared['tot_samples_loaded'] = self.tot_samples_loaded
        return shared

    def _setup_data(self, datatype):
        """
        Passthrough.
        """
        pass

    def num_episodes(self):
        if self.is_train:
            return self.num_eps
        else:
            return self.num_eps // self.dws + int((self.num_eps % self.dws) > self.rank)

    def num_examples(self):
        if self.is_train:
            return self.num_exs
        else:
            return self.num_exs // self.dws + int((self.num_exs % self.dws) > self.rank)

    def next_episode_idx(self):
        # We don't actually track episodes in ChunkTeacher, we just blindly
        # trust the queue. This hacks around FixedDialogTeacher's next_example
        # check that the epoch is done.
        return 0

    def _enqueue_request(self):
        """
        Queue a request for loading to the data loader.
        """
        if self.threading:
            self.data_loader.request_load(self.receive_data, self.get_chunk, ())
        else:
            self._process_data(self.get_chunk())

    def receive_data(self, future):
        """
        Receive loaded data and place it onto the sample queue.

        :param future:
            A Future object which will return a value from a call to get_chunk()
        """
        return self._process_data(future.result())

    def _process_data(self, output: Optional[Tuple[Any, int]]):
        """
        Loads data.

        Load data into self.samples until buffersize is reached.

        :param output:
            The output of an item from a call to get_chunk()
        """
        if output is None:
            return
        chunk_output, chunk_reset_cnt = output
        if chunk_output is None:
            self.samples.put((None, chunk_reset_cnt))
            return
        if self.threading:
            while chunk_output:
                # self.samples is a queue with maxsize
                # self.buffersize, so will block if the
                # buffer gets full
                sample = chunk_output.pop(0)
                if (
                    self.is_train
                    or self.tot_samples_loaded[chunk_reset_cnt] % self.dws == self.rank
                ):
                    # log the reset count at the time the chunk was queued
                    self.samples.put((sample, chunk_reset_cnt))
                self.tot_samples_loaded[chunk_reset_cnt] += 1
        else:
            # we're actually running in single processor mode so we'll just
            # do a thread-unsafe hit of the python internals, which is much faster
            # than trying to safely put things onto the queue
            self.samples.queue.extend((co, chunk_reset_cnt) for co in chunk_output)
            self.tot_samples_loaded[chunk_reset_cnt] += len(chunk_output)
        if self.threading:
            # and start loading the next chunk
            self._enqueue_request()

    def _enqueue_chunks(self):
        """
        Shuffles and queues fold chunks for loading.
        """
        if DatatypeHelper.should_shuffle(self.datatype):
            self.rng.shuffle(self.fold_chunks)
        # save the reset count at the time a chunk was queued
        reset_cnt = self.reset_counter.value()
        for c in self.fold_chunks:
            self.chunks.put((c, reset_cnt))
        self.chunks.put((None, reset_cnt))
        # gross hack: in training models, when we get to validation, we enqueue
        # a request in the constructor, followed by another enqueue from a
        # reset immediately after. If the former is already running, we'll end
        # up with one too many calls to get_chunk and block on termination.
        # That's what I refer to as "losing" the race condition. If you look in
        # get_chunk, you'll also find the spot where we "win" the race
        # condition.
        self.chunks.put((None, reset_cnt))

    @abstractmethod
    def load_from_chunk(self, chunk_idx: int) -> List[ChunkOutput]:
        """
        [Abstract] Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        pass

    @abstractmethod
    def create_message(self, queue_output: ChunkOutput, entry_idx=0) -> 'Message':
        """
        [Abstract] Given the tuple output of the queue, return an act.

        May depend on entry index if queue output is a multi-turn episode.
        """
        pass

    def get_chunk(self):
        """
        Refill the buffer.
        """
        next_chunk, chunk_reset_cnt = self.chunks.get()
        if next_chunk is None:
            if DatatypeHelper.should_cycle(self.datatype):
                # start putting chunks back onto the queue
                self._enqueue_chunks()
                next_chunk, chunk_reset_cnt = self.chunks.get()
                if next_chunk is None:
                    # See the race condition described around "gross hack" in
                    # _enqueue_chunks.  if we win the race condition, then
                    # catch it here
                    next_chunk, chunk_reset_cnt = self.chunks.get()
            else:
                # if we're in valid/test, we need to actually signal the end
                return (None, chunk_reset_cnt)
        # abstract method `load_from_chunk` returns a list of tuples
        output = self.load_from_chunk(next_chunk)

        if DatatypeHelper.should_shuffle(self.datatype):
            # randomize the samples
            random.Random().shuffle(output)
        return output, chunk_reset_cnt

    def next_example(self):
        # next_example will always fail to provide useful signal on whether
        # we're at the end of an epoch in chunk teacher. Instead, the queue
        # empties and we simply start outputting pads forever. As such, we'll
        # measure epochs when we start receiving only pads.

        # (This isn't relevant for the training loop, which loops for ever and
        # never "epochs").
        retval, fake_epoch_done = super().next_example()
        real_epoch_done = retval.is_padding()
        return retval, real_epoch_done

    def get(self, episode_idx, entry_idx=0):
        if not self.threading and self.samples.empty():
            self._enqueue_request()

        curr_reset_cnt = self.reset_counter.value()
        if self._episode_done:
            # Get the next episode or example
            queue_output, reset_cnt = self.samples.get()
            stale_exs = 0
            while curr_reset_cnt > reset_cnt:
                stale_exs += 1
                queue_output, reset_cnt = self.samples.get()
            if stale_exs > 0:
                logging.debug(f"Removed {stale_exs} stale examples from the queue.")
            if queue_output is None:
                self.samples.put((None, reset_cnt))
                return Message.padding_example()

            # Update the last queue output in the case
            # of multi-turn episodes
            self.last_queue_output = queue_output

        # create a Message object from the queue output
        msg = self.create_message(self.last_queue_output, entry_idx)
        self._episode_done = msg['episode_done']

        return msg

    def _drain(self, q):
        while not q.empty():
            try:
                q.get()
            except queue.Empty:
                return

    def reset(self):
        super().reset()
        if self.is_root_teacher:
            self.reset_counter.increment()
            # drain the queues and refill the chunk queue with a new epoch.
            # additionally, we have to relaunch the loader
            self._drain(self.samples)
            self._drain(self.chunks)
            self._enqueue_chunks()
            self.tot_samples_loaded.clear()  # reset the count of samples loaded
            if self.threading:
                self._enqueue_request()

    def shutdown(self):
        # Time to wrap up. We should rush out to the worker and tell them
        # that they're "done" processing data.
        # same signal as end of epoch
        self.chunks.put((None, None))
        self.chunks.put((None, None))


def _add_task_flags_to_agent_opt(agent, opt: Opt, flags):
    """
    Handle task flags provided by the task name itself.

    With this you can set specific opts with `-t task:flag=foo`.
    """
    fl = flags.split(':')
    task = []
    for f in fl:
        if '=' in f:
            one_flag = f.split('=')
            key = one_flag[0].replace('-', '_')
            raw_value = one_flag[1].replace(';', ':')

            # Convert to bool/int/float if necessary
            if raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(raw_value)  # type: ignore
                except ValueError:
                    try:
                        value = float(raw_value)  # type: ignore
                    except ValueError:
                        value = raw_value  # type: ignore

            opt[key] = value
        else:
            task.append(f)
    opt['task'] = ':'.join(task)


def create_task_agent_from_taskname(opt: Opt):
    """
    Create task agent(s) assuming the input ``task_dir:teacher_class``.

    e.g. def_string is a shorthand path like ``babi:Task1k:1`` or ``#babi`` or a
    complete path like ``parlai.tasks.babi.agents:Task1kTeacher:1``, which essentially
    performs ``from parlai.tasks.babi import Task1kTeacher`` with the parameter ``1`` in
    ``opt['task']`` to be used by the class ``Task1kTeacher``.
    """
    if not opt.get('task'):
        raise RuntimeError(
            'No task specified. Please select a task with ' + '--task {task_name}.'
        )
    if ',' not in opt['task']:
        # Single task
        teacher_class = load_teacher_module(opt['task'])
        _add_task_flags_to_agent_opt(teacher_class, opt, opt['task'])
        task_agents = teacher_class(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents
    else:
        # Multitask teacher/agent
        task_agents = MultiTaskTeacher(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents

# from parlai.core.teachers import create_task_agent_from_taskname
# from parlai.utils.data import DatatypeHelper


#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities related to handling data.
"""
import random
from typing import List


class DatatypeHelper:
    """
    Helper class to determine properties from datatype strings.
    """

    @classmethod
    def fold(cls, datatype: str) -> str:
        """
        Extract the fold part of the datatype.

        :param datatype:
            parlai datatype

        :return: the fold

        >>> DatatypeHelper.fold("train:ordered")
        ... "train"
        """
        return datatype.split(':')[0]

    @classmethod
    def strip_stream(cls, datatype: str) -> str:
        """
        Remove :stream from the datatype.

        Used by ChunkTeacher where behavior does not change based on streaming.

        :param datatype:
            parlai datatype

        :return:
            a non-streaming version of the datatype.

        >>> DatatypeHelper.fold("train:stream")
        "train"
        >>> DatatypeHelper.fold("train")
        "train"
        """
        return datatype.replace(":stream", "")

    @classmethod
    def should_cycle(cls, datatype: str) -> bool:
        """
        Return whether we should cycle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_cycle:
            given datatype, return whether we should cycle
        """
        assert datatype is not None, 'datatype must not be none'
        return (
            'train' in datatype
            and 'evalmode' not in datatype
            and 'ordered' not in datatype
        )

    @classmethod
    def should_shuffle(cls, datatype: str) -> bool:
        """
        Return whether we should shuffle data based on the datatype.

        :param datatype:
            parlai datatype

        :return should_shuffle:
            given datatype, return whether we should shuffle
        """
        assert datatype is not None, 'datatype must not be none'
        return (
            'train' in datatype
            and 'evalmode' not in datatype
            and 'ordered' not in datatype
            and 'stream' not in datatype
        )

    @classmethod
    def is_training(cls, datatype: str) -> bool:
        """
        Return whether we should return eval_labels or labels.

        :param datatype:
            parlai datatype

        :return is_training:
            bool indicating whether should return eval_labels or labels
        """
        assert datatype is not None, 'datatype must not be none'
        return 'train' in datatype and 'evalmode' not in datatype

    @classmethod
    def is_streaming(cls, datatype: str) -> bool:
        """
        Return whether this is streaming.

        :param datatype:
            parlai datatype

        :returns:
            bool indicating whether we are streaming
        """
        return 'stream' in datatype

    @classmethod
    def split_data_by_fold(
        cls,
        fold: str,
        data: List,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int = 42,
    ):
        """
        Splits a list of data into train/valid/test folds. The members of these folds
        are randomized (in a consistent manner) by a seed. This is a convenience
        function for datasets that do not have a canonical split.

        :param fold:
           parlai fold/datatype
        :param data:
            List of data examples to be split
        :param train_frac:
            Fraction of data to be used for the "train" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param valid_frac:
            Fraction of data to be used for the "valid" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param test_frac:
            Fraction of data to be used for the "test" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param seed:
            Seed for shuffling
        """
        assert train_frac + valid_frac + test_frac == 1
        if "train" in fold:
            start = 0.0
            end = train_frac
        elif "valid" in fold:
            start = train_frac
            end = train_frac + valid_frac
        else:
            start = train_frac + valid_frac
            end = 1.0

        random.Random(seed).shuffle(data)
        return data[int(start * len(data)) : int(end * len(data))]

    @classmethod
    def split_subset_data_by_fold(
        cls,
        fold: str,
        subsets: List[List],
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int = 42,
    ):
        """
        Splits a list of subsets of data, where we want equal samples from each subset,
        into train/valid/test folds, ensuring that samples from a given subset are not
        changed to another fold as more subsets are added.

        For example, say a dataset has domains A, B. Let's say we have an experiment where we train and validate a model on domain A, then on domains A + B. If we naively concatinate the subset of data from A + B together, randomize it, and split the result into train, valid, and test folds, there is no guarantee that valid or test examples from A-only will not end up into the train fold of the A + B split from this naive concatination process.

        The members of these folds are randomized (but in a fixed manner) by a seed.

        :param fold:
           parlai fold/datatype
        :param subsets:
            List of subsets of data examples to be split
        :param train_frac:
            Fraction of data to be used for the "train" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param valid_frac:
            Fraction of data to be used for the "valid" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param test_frac:
            Fraction of data to be used for the "test" fold. train_frac, valid_frac, and test_frac should sum to 1.
        :param seed:
            Seed for shuffling
        """
        result = []
        for subset in subsets:
            result.extend(
                cls.split_data_by_fold(
                    fold, subset, train_frac, valid_frac, test_frac, seed
                )
            )
        random.Random(seed).shuffle(result)
        return result



# from parlai.utils.misc import Timer, display_messages, warn_once
# from parlai.tasks.tasks import ids_to_tasks
# from parlai.utils.misc import error_once


def validate(observation):
    """
    Make sure the observation table is valid, or raise an error.
    """
    if observation is not None and isinstance(observation, dict):
        return observation
    else:
        raise RuntimeError('Must return dictionary or Message object from act().')


class World(object):
    """
    Empty parent providing null definitions of API functions for Worlds.

    All children can override these to provide more detailed functionality.
    """

    def __init__(self, opt: Opt, agents=None, shared=None):
        self.id = opt['task']
        self.opt = copy.deepcopy(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents to world directly.
            self.agents = agents
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time = Timer()

    def parley(self):
        """
        Perform one step of actions for the agents in the world.

        This is empty in the base class.
        """
        # TODO: mark as abstract?
        pass

    def getID(self):
        """
        Return the name of the world, typically the task the world encodes.
        """
        return self.id

    def display(self):
        """
        Return a string describing the current state of the world.

        Useful for monitoring and debugging. By default, display the messages between
        the agents.
        """
        if not hasattr(self, 'acts'):
            return ''
        return display_messages(
            self.acts,
            ignore_agent_reply=self.opt.get('ignore_agent_reply', False),
            add_fields=self.opt.get('display_add_fields', ''),
            prettify=self.opt.get('display_prettify', False),
            max_len=self.opt.get('max_display_len', 1000),
            verbose=self.opt.get('verbose', False),
        )

    def episode_done(self):
        """
        Whether the episode is done or not.
        """
        return False

    def epoch_done(self):
        """
        Whether the epoch is done or not.

        Not all worlds have the notion of an epoch, but this is useful for fixed
        training, validation or test sets.
        """
        return False

    def share(self):
        """
        Share the world.
        """
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['agents'] = self._share_agents()
        return shared_data

    def clone(self):
        """
        Create a duplicate of the world.
        """
        return type(self)(opt=self.opt, agents=None, shared=self.share())

    def _share_agents(self):
        """
        Create shared data for agents.

        Allows other classes to create the same agents without duplicating the data
        (i.e. sharing parameters).
        """
        if not hasattr(self, 'agents'):
            return None
        shared_agents = [a.share() for a in self.agents]
        return shared_agents

    def get_agents(self):
        """
        Return the list of agents.
        """
        return self.agents

    def get_task_agent(self):
        """
        Return task agent, if applicable.
        """
        raise NotImplementedError('Implement in subworld')

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        raise NotImplementedError('Implement in subworld')

    def get_acts(self):
        """
        Return the last act of each agent.
        """
        return self.acts

    def get_time(self):
        """
        Return total training time.
        """
        return self.time.time()

    def get_total_exs(self):
        """
        Return total amount of examples seen by world.
        """
        return self.total_exs

    def get_total_epochs(self):
        """
        Return total amount of epochs on which the world has trained.
        """
        return self.total_epochs

    def get_total_parleys(self):
        """
        Return total number of parleys.
        """
        return self.total_parleys

    def __enter__(self):
        """
        Empty enter provided for use with ``with`` statement.

        e.g:

        .. code-block:: python

            with World() as world:
                for n in range(10):
                    n.parley()
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        After ``with`` statement, call shutdown.
        """
        self.shutdown()
        return False

    def num_examples(self):
        """
        Return the number of examples.

        Always 0 in the abstract world.
        """
        # TODO: mark as abstract?
        return 0

    def num_episodes(self):
        """
        Return the number of episodes.

        Always 0 in the abstract world.
        """
        # TODO: mark as abstract?
        return 0

    def reset(self):
        """
        Reset all agents in the world, and world statistics.
        """
        self.reset_agents()
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time.reset()

    def reset_agents(self):
        """
        Reset all agents in the world.
        """
        agents = self.get_agents()
        for a in agents:
            a.reset()

    def reset_metrics(self):
        """
        Reset metrics for all agents.
        """
        for a in self.agents:
            a.reset_metrics()

    def shutdown(self):
        """
        Perform any cleanup, if appropriate.
        """
        pass

    def update_counters(self):
        """
        Update how many epochs have completed.
        """
        self.total_parleys += 1
        if self.max_exs is None:
            if 'num_epochs' in self.opt and self.opt['num_epochs'] > 0:
                if self.num_examples:
                    self.max_exs = self.num_examples() * self.opt['num_epochs']
                else:
                    self.max_exs = -1
            else:
                self.max_exs = -1
        # when we know the size of the data
        if self.max_exs > 0 or self.num_examples():
            self.total_epochs = (
                self.total_parleys * self.opt.get('batchsize', 1) / self.num_examples()
            )
        # when we do not know the size of the data
        else:
            if self.epoch_done():
                self.total_epochs += 1


class DialogPartnerWorld(World):
    """
    Simple world for two agents communicating synchronously.

    This basic world switches back and forth between two agents, giving each agent one
    chance to speak per turn and passing that back to the other one.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Return the parser as-is.

        Self-chat-specific world flags can be added here.
        """
        return parser

    def __init__(self, opt: Opt, agents=None, shared=None):
        if not ((agents is not None) ^ (shared is not None)):
            raise ValueError('You must supply either agents or shared, but not both.')
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            if len(agents) != 2:
                raise RuntimeError('There must be exactly two agents for this world.')
            # Add passed in agents directly.
            self.agents = agents
        self.acts = [None] * len(self.agents)
        if self.agents is not None and len(self.agents) > 0:
            # Name the world after the first agent.
            self.id = self.get_task_agent().getID()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        return self.get_agents()[1]

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()

    def episode_done(self):
        """
        Only the first agent indicates when the episode is done.
        """
        if self.acts[0] is not None:
            return self.acts[0].get('episode_done', False)
        else:
            return False

    def epoch_done(self):
        """
        Only the first agent indicates when the epoch is done.
        """
        return self.get_task_agent().epoch_done()

    def report(self):
        """
        Report all metrics of all subagents.
        """
        # from parlai.core.metrics import Metric, LegacyMetric

        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if not isinstance(v, Metric):
                        v = LegacyMetric(v)
                    if k not in metrics:
                        # first agent gets priority in settings values for keys
                        # this way model can't e.g. override accuracy to 100%
                        metrics[k] = v
        if metrics and 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def num_examples(self):
        """
        Return number of examples.
        """
        if hasattr(self.get_task_agent(), 'num_examples'):
            return self.get_task_agent().num_examples()
        return 0

    def num_episodes(self):
        """
        Return number of episodes.
        """
        if hasattr(self.get_task_agent(), 'num_episodes'):
            return self.get_task_agent().num_episodes()
        return 0

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()

    def update_counters(self):
        """
        Ensure all counters are synchronized across threads.
        """
        super().update_counters()
        for a in self.agents:
            if hasattr(a, 'update_counters'):
                a.update_counters()


class MultiAgentDialogWorld(World):
    """
    Basic world where each agent gets a turn in a round-robin fashion.

    Each agent receives as input the actions of all other agents since its last `act()`.
    """

    def __init__(self, opt: Opt, agents, shared=None):
        super().__init__(opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            # Add passed in agents directly.
            self.agents = agents
        self.acts = [None] * len(self.agents)

    def parley(self):
        """
        Perform a turn for every agent.

        For each agent, get an observation of the last action each of the other agents
        took. Then take an action yourself.
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            acts[index] = agent.act()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        self.update_counters()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return self.get_agents()[0]

    def get_model_agent(self):
        """
        Return model agent.
        """
        return self.get_agents()[1]

    def epoch_done(self):
        """
        Return if the epoch is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.epoch_done():
                done = True
        return done

    def episode_done(self):
        """
        Return if the episode is done for any subagent.
        """
        done = False
        for a in self.agents:
            if a.episode_done():
                done = True
        return done

    def report(self):
        """
        Report metrics for all subagents.
        """
        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if k not in metrics:
                        # first agent gets priority in settings values for keys
                        # this way model can't e.g. override accuracy to 100%
                        metrics[k] = v
        if metrics and 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()


class MultiWorld(World):
    """
    Container for multiple worlds.

    Container for a set of worlds where each world gets a turn in a round-robin fashion.
    The same user_agents are placed in each, though each world may contain additional
    agents according to the task that world represents.
    """

    def __init__(self, opt: Opt, agents=None, shared=None, default_world=None):
        super().__init__(opt)
        self.worlds: List[World] = []
        for index, k in enumerate(opt['task'].split(',')):
            k = k.strip()
            if k:
                if shared:
                    # Create worlds based on shared data.
                    s = shared['worlds'][index]
                    self.worlds.append(s['world_class'](s['opt'], None, s))
                else:
                    # Agents are already specified.
                    opt_singletask = copy.deepcopy(opt)
                    opt_singletask['task'] = k
                    self.worlds.append(
                        create_task_world(
                            opt_singletask, agents, default_world=default_world
                        )
                    )
        self.world_idx = -1
        self.new_world = True
        self.parleys = -1
        # Check to see if we are training
        self.is_training = DatatypeHelper.is_training(opt.get('datatype'))
        # Check to see if we should shuffle
        self.should_shuffle = DatatypeHelper.should_shuffle(opt.get('datatype'))
        # Make multi-task task probabilities.
        self.cum_task_weights = [1] * len(self.worlds)
        self.task_choices = range(len(self.worlds))
        weights = self.opt.get('multitask_weights', [1])
        # Warn about multi-task weights being ignored if we are in a datatype that doesn't involve shuffling
        if weights != [1] and not self.should_shuffle:
            warn_once(
                f"WARNING: multitask weights are ignored for datatype {opt.get('datatype')} as we iterate through tasks in a round robin"
            )
        if weights == 'stochastic':
            weights = [w.num_episodes() for w in self.worlds]
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight
        task_ids: Dict[str, Teacher] = {}
        # Having overlap in teacher ids will cause issues for metrics aggregation.
        for each_world in self.worlds:
            world_id = each_world.getID()
            if world_id in task_ids:
                world_class = each_world.get_agents()[0].__class__
                error_once(
                    f"{task_ids[world_id]} and {world_class} teachers have overlap "
                    f"in id '{world_id}'. This will cause their metrics to be "
                    "intermingled. Change the id attribute of one to remove this "
                    "message."
                )
            else:
                task_ids[world_id] = each_world.get_task_agent()

    def num_examples(self):
        """
        Return sum of each subworld's number of examples.
        """
        if not hasattr(self, 'num_exs'):
            worlds_num_exs = [w.num_examples() for w in self.worlds]
            if any(num is None for num in worlds_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(worlds_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return sum of each subworld's number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            worlds_num_eps = [w.num_episodes() for w in self.worlds]
            if any(num is None for num in worlds_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(worlds_num_eps)
        return self.num_eps

    def get_agents(self):
        """
        Return the agents in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_agents()

    def get_task_agent(self):
        """
        Not possible/well-defined in this setting.
        """
        return self.worlds[self.world_idx].get_task_agent()

    def get_model_agent(self):
        """
        Not implemented.
        """
        return self.worlds[self.world_idx].get_model_agent()

    def get_acts(self):
        """
        Return the acts in the *current* subworld.
        """
        return self.worlds[self.world_idx].get_acts()

    def share(self):
        """
        Share all the subworlds.
        """
        shared_data = {}
        shared_data['world_class'] = type(self)
        shared_data['opt'] = self.opt
        shared_data['worlds'] = [w.share() for w in self.worlds]
        return shared_data

    def epoch_done(self):
        """
        Return if *all* the subworlds are done.
        """
        for t in self.worlds:
            if not t.epoch_done():
                return False
        return True

    def parley_init(self):
        """
        Update the current subworld.

        If we are in the middle of an episode, keep the same world and finish this
        episode. If we have finished this episode, pick a new world (either in a random
        or round-robin fashion).
        """
        self.parleys = self.parleys + 1
        if self.world_idx >= 0 and self.worlds[self.world_idx].episode_done():
            self.new_world = True
        if self.new_world:
            self.new_world = False
            self.parleys = 0
            if self.should_shuffle:
                # select random world
                self.world_idx = random.choices(
                    self.task_choices, cum_weights=self.cum_task_weights
                )[0]
            else:
                # do at most one full loop looking for unfinished world
                for _ in range(len(self.worlds)):
                    self.world_idx = (self.world_idx + 1) % len(self.worlds)
                    if not self.worlds[self.world_idx].epoch_done():
                        # if this world has examples ready, break
                        break

    def parley(self):
        """
        Parley the *current* subworld.
        """
        self.parley_init()
        self.worlds[self.world_idx].parley()
        self.update_counters()

    def display(self):
        """
        Display all subworlds.
        """
        if self.world_idx != -1:
            s = ''
            w = self.worlds[self.world_idx]
            if self.parleys == 0:
                s = '[world ' + str(self.world_idx) + ':' + w.getID() + ']\n'
            s = s + w.display()
            return s
        else:
            return ''

    def report(self):
        """
        Report aggregate metrics across all subworlds.
        """
        metrics = aggregate_named_reports(
            {w.getID(): w.report() for w in self.worlds},
            micro_average=self.opt.get('aggregate_micro', False),
        )
        if 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def reset(self):
        """
        Reset all subworlds.
        """
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in all subworlds.
        """
        for w in self.worlds:
            w.reset_metrics()

    def update_counters(self):
        super().update_counters()
        for w in self.worlds:
            w.update_counters()


def _override_opts_in_shared(table, overrides):
    """
    Override all shared dicts.

    Looks recursively for ``opt`` dictionaries within shared dict and overrides any key-
    value pairs with pairs from the overrides dict.
    """
    if 'opt' in table:
        # change values if an 'opt' dict is available
        for k, v in overrides.items():
            table['opt'][k] = v
    for k, v in table.items():
        # look for sub-dictionaries which also might contain an 'opt' dict
        if type(v) == dict and k != 'opt' and 'opt' in v:
            _override_opts_in_shared(v, overrides)
        elif type(v) == list:
            for item in v:
                if type(item) == dict and 'opt' in item:
                    # if this is a list of agent shared dicts, we want to iterate
                    _override_opts_in_shared(item, overrides)
                else:
                    # if this is e.g. list of candidate strings, stop right away
                    break
    return table


class BatchWorld(World):
    """
    BatchWorld contains many copies of the same world.

    Create a separate world for each item in the batch, sharing
    the parameters for each.

    The underlying world(s) it is batching can be either
    ``DialogPartnerWorld``, ``MultiAgentWorld``, or ``MultiWorld``.
    """

    def __init__(self, opt: Opt, world):
        super().__init__(opt)
        self.opt = opt
        self.random = opt.get('datatype', None) == 'train'
        self.world = world
        self.worlds: List[World] = []
        for i in range(opt['batchsize']):
            # make sure that any opt dicts in shared have batchindex set to i
            # this lets all shared agents know which batchindex they have,
            # which is needed for ordered data (esp valid/test sets)
            shared = world.share()
            shared['batchindex'] = i
            for agent_shared in shared.get('agents', ''):
                agent_shared['batchindex'] = i
            # TODO: deprecate override_opts
            _override_opts_in_shared(shared, {'batchindex': i})
            self.worlds.append(shared['world_class'](opt, None, shared))
        self.batch_observations = [None] * len(self.world.get_agents())
        self.first_batch = None
        self.acts = [None] * len(self.world.get_agents())

    def batch_observe(self, index, batch_actions, index_acting):
        """
        Observe corresponding actions in all subworlds.
        """
        batch_observations = []
        for i, w in enumerate(self.worlds):
            agents = w.get_agents()
            observation = None
            if batch_actions[i] is None:
                # shouldn't send None, should send empty observations
                batch_actions[i] = [{}] * len(self.worlds)

            if hasattr(w, 'observe'):
                # The world has its own observe function, which the action
                # first goes through (agents receive messages via the world,
                # not from each other).
                observation = w.observe(agents[index], validate(batch_actions[i]))
            else:
                observation = validate(batch_actions[i])

            if index == index_acting:
                # self_observe is distinguished from a normal observe
                if hasattr(agents[index], 'self_observe'):
                    agents[index].self_observe(observation)
            else:
                observation = agents[index].observe(observation)

            # TODO: not so sure about this...
            if observation is None:
                raise ValueError('Agents should return what they observed.')
            batch_observations.append(observation)
        return batch_observations

    def batch_act(self, agent_idx, batch_observation):
        """
        Act in all subworlds.
        """
        # Given batch observation, do update for agents[index].
        # Call update on agent
        a = self.world.get_agents()[agent_idx]
        if hasattr(a, 'batch_act'):
            batch_actions = a.batch_act(batch_observation)
            # Store the actions locally in each world.
            for i, w in enumerate(self.worlds):
                acts = w.get_acts()
                acts[agent_idx] = batch_actions[i]
        else:
            # Reverts to running on each individually.
            batch_actions = []
            for w in self.worlds:
                agents = w.get_agents()
                acts = w.get_acts()
                acts[agent_idx] = agents[agent_idx].act()
                batch_actions.append(acts[agent_idx])
        return batch_actions

    def parley(self):
        """
        Parley in all subworlds.

        Usually with ref:`batch_act` and ref:`batch_observe`.
        """
        # Collect batch together for each agent, and do update.
        # Assumes DialogPartnerWorld, MultiAgentWorld, or MultiWorlds of them.
        num_agents = len(self.world.get_agents())
        batch_observations = self.batch_observations

        if hasattr(self.world, 'parley_init'):
            for w in self.worlds:
                w.parley_init()

        for agent_idx in range(num_agents):
            # The agent acts.
            batch_act = self.batch_act(agent_idx, batch_observations[agent_idx])
            self.acts[agent_idx] = batch_act
            # We possibly execute this action in the world.
            if hasattr(self.world, 'execute'):
                for w in self.worlds:
                    w.execute(w.agents[agent_idx], batch_act[agent_idx])
            # All agents (might) observe the results.
            for other_index in range(num_agents):
                obs = self.batch_observe(other_index, batch_act, agent_idx)
                if obs is not None:
                    batch_observations[other_index] = obs
        self.update_counters()

    def display(self):
        """
        Display the full batch.
        """
        s = "[--batchsize " + str(len(self.worlds)) + "--]\n"
        for i, w in enumerate(self.worlds):
            s += "[batch world " + str(i) + ":]\n"
            s += w.display() + '\n'
        s += "[--end of batch--]"
        return s

    def num_examples(self):
        """
        Return the number of examples for the root world.
        """
        return self.world.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes for the root world.
        """
        return self.world.num_episodes()

    def get_total_exs(self):
        """
        Return the total number of processed episodes in the root world.
        """
        return self.world.get_total_exs()

    def getID(self):
        """
        Return the ID of the root world.
        """
        return self.world.getID()

    def get_agents(self):
        """
        Return the agents of the root world.
        """
        return self.world.get_agents()

    def get_task_agent(self):
        """
        Return task agent of the root world.
        """
        return self.world.get_task_agent()

    def get_model_agent(self):
        """
        Return model agent of the root world.
        """
        return self.world.get_model_agent()

    def episode_done(self):
        """
        Return whether the episode is done.

        A batch world is never finished, so this always returns `False`.
        """
        return False

    def epoch_done(self):
        """
        Return if the epoch is done in the root world.
        """
        # first check parent world: if it says it's done, we're done
        if self.world.epoch_done():
            return True
        # otherwise check if all shared worlds are done
        for world in self.worlds:
            if not world.epoch_done():
                return False
        return True

    def report(self):
        """
        Report metrics for the root world.
        """
        return self.world.report()

    def reset(self):
        """
        Reset the root world, and all copies.
        """
        self.world.reset()
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        """
        Reset metrics in the root world.
        """
        self.world.reset_metrics()

    def shutdown(self):
        """
        Shutdown each world.
        """
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for dealing with torch code.
"""

import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
# import parlai.utils.logging as logging
# import parlai.utils.io as io_utils


try:
    import torch
except ImportError:
    raise ImportError('Parlai requires pytorch. Go to http://pytorch.org to install.')

import torch.optim

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

# according to the tensor cores documentation from nvidia, the matmuls in fp16
# must all be multiples of 8 in order to get the speedup from fp16. We set this
# as a constant here for clarity and convenience.  See
# https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/ for more
# information.
FP16_PAD_SIZE = 8


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def atomic_save(state_dict: Any, path: str) -> None:
    """
    Like torch.save, but atomic.

    Useful for preventing trouble coming from being pre-empted or killed while writing
    to disk. Works by writing to a temporary file, and then renaming the file to the
    final name.
    """

    if USE_ATOMIC_TORCH_SAVE:
        with open(path + ".tmp", "wb") as f:
            torch.save(state_dict, f)
        os.replace(path + ".tmp", path)
    else:
        with PathManager.open(path, "wb") as f:
            torch.save(state_dict, f)


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    left_padded: bool = False,
    max_len: Optional[int] = None,
    fp16friendly: bool = False,
) -> Tuple[torch.LongTensor, List[int]]:
    """
    Create a padded matrix from an uneven list of lists.

    Returns (padded, lengths), where padded is the padded matrix, and lengths
    is a list containing the lengths of each row.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param bool sort: If True, orders by the length
    :param int pad_idx: the value to use for padding
    :param bool left_padded:
    :param int max_len: if None, the max length is the maximum item length
    :param bool fp16friendly: if True, pads the time dimension to be a multiple of 4.

    :returns: (padded, lengths) tuple
    :rtype: (Tensor[int64], list[int])
    """

    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len

    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if fp16friendly and (t % FP16_PAD_SIZE != 0):
        # pad to be fp16 friendly
        t += FP16_PAD_SIZE - (t % FP16_PAD_SIZE)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.LongTensor(item)  # type: ignore
        if left_padded:
            # place at end
            output[i, t - length :] = item
        else:
            # place at beginning
            output[i, :length] = item

    return output, lens


def padded_3d(
    tensors: List[torch.LongTensor],
    pad_idx: int = 0,
    dtype: Optional[torch.dtype] = torch.long,
    fp16friendly: bool = False,
):
    """
    Make 3D padded tensor for list of lists of 1D tensors or lists.

    Will keep items on the same device as originally.

    :param tensors:
        list of lists of 1D tensors (or lists)
    :param pad_idx:
        padding to fill tensor with
    :param bool fp16friendly:
        if True, pads the final dimension to be a multiple of 8.

    :returns:
        3D tensor with the maximum dimensions of the inputs
    """
    a = len(tensors)
    b = max(len(row) for row in tensors)  # type: ignore
    c = max(len(item) for row in tensors for item in row)  # type: ignore

    # pad empty tensors
    if fp16friendly and c % FP16_PAD_SIZE != 0:
        c += FP16_PAD_SIZE - (c % FP16_PAD_SIZE)
    c = max(c, 1)

    dev = tensors[0][0].device
    output = torch.full((a, b, c), pad_idx, dtype=dtype, device=dev)

    for i, row in enumerate(tensors):
        item: Sized
        for j, item in enumerate(row):  # type: ignore
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.as_tensor(item, dtype=dtype)
            output[i, j, : len(item)] = item

    return output


def concat_without_padding(text_idx, cand_idx, use_cuda, null_idx=0):
    """
    Concatenate two right padded tensors and move padding to the right.

    For example,
        if text_idx = [[1, 2, 3, 4, 0, 0  ]]
        and cand_idx = [[5, 6, 7, 8, 0, 0 ]]:
    Then result = (tokens, segments) where
        tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
        segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
    """
    assert text_idx.size(0) == cand_idx.size(0)
    assert len(text_idx.size()) == 2
    assert len(cand_idx.size()) == 2
    segments_idx = [0, 1]
    text_idx = text_idx.cpu()
    cand_idx = cand_idx.cpu()
    cand_len = cand_idx.size(1)
    concat_len = text_idx.size(1) + cand_idx.size(1)
    tokens = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    segments = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    for i in range(len(tokens)):
        non_nuls = torch.sum(text_idx[i, :] != null_idx)
        tokens[i, 0:non_nuls] = text_idx[i, 0:non_nuls]
        segments[i, 0:non_nuls] = segments_idx[0]
        tokens[i, non_nuls : non_nuls + cand_len] = cand_idx[i, :]
        segments[i, non_nuls : non_nuls + cand_len] = segments_idx[1]
    if use_cuda:
        tokens = tokens.cuda()
        segments = segments.cuda()
    return tokens, segments


def argsort(keys: List[Any], *lists: List[List[Any]], descending: bool = False):
    """
    Reorder each list in lists by the (descending) sorted order of keys.

    :param iter keys:
        Keys to order by.
    :param list[list] lists:
        Lists to reordered by keys's order.  Correctly handles lists and 1-D
        tensors.
    :param bool descending:
        Use descending order if true.

    :returns:
        The reordered items.
    """
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        # watch out in case we don't have torch installed
        if isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output


def compute_grad_norm(parameters, norm_type=2.0):
    """
    Compute norm over gradients of model parameters.

    :param parameters:
        the model parameters for gradient norm calculation. Iterable of
        Tensors or single Tensor
    :param norm_type:
        type of p-norm to use

    :returns:
        the computed gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None and p.grad is not None]
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


class IdentityLayer(torch.nn.Module):
    """
    Identity layer module.

    Useful for decoder-only Torch Generator agents.
    """

    def forward(self, xs):
        """
        Identity.
        """
        return xs


def total_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of parameters in the model.

    :param model:
        the model whose parameters we wish to count.

    :return:
        total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in the model.

    :param model:
        the model whose parameters we wish to count.

    :return:
        total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


Chunk = TypeVar('Chunk')


PipelineWorkItem = namedtuple(
    'PipelineWorkItem', ['chunk_idx', 'layer_nos', 'next_device']
)


class PipelineHelper(object):
    """
    PipelineHelper assists with implementing pipelining in model parallelism.

    For a tutorial on model parallelism, as it's implemented in parts of ParlAI,
    see https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html.

    Usage:
    >>> my_model = PipelineHelper().make_parallel(my_model)

    Note that you will need to manually implement logic which handles the
    moved layers.
    """

    def __init__(self):
        self.__device_allocations = {}
        self.num_devices = torch.cuda.device_count()
        self.devices = []
        for i in range(self.num_devices):
            d = f'cuda:{i}'
            self.devices.append(d)
            self.__device_allocations[d] = 0

    def check_compatibility(self, opt):
        """
        Check compatibility for opts.

        Really just used to raise an error message if the user mixes multiprocessing and
        model parallelism.
        """
        if opt.get('multiprocessing') and not os.environ.get('PARLAI_FORCE_MP'):
            raise RuntimeError(
                "It looks like you are trying to mix multiprocessing data "
                "parallelism (multiprocessing_train or multiprocessing_eval) "
                "with --model-parallel true. This is almost certainly a user "
                "error, and is going to result in hanging as the two methods "
                "fight for resources. Use simple `train_model` instead of "
                "`mp_train`, or add `--model-parallel false`. For more info, "
                "see https://github.com/facebookresearch/ParlAI/issues/2962."
            )

    def make_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Allocate specific layers in a model to be ModelParallel.

        Limited to only ModuleLists within the model.  Uses some heuristics to
        attempt to evenly distribute layers across GPUs, in order to balance
        memory usage. They are:

        - Assume the 0th GPU will host the optimizer, word embeddings, etc.
        - Assume activation memory is linear with the number of parameters.
        - All layers are approximately equal in size.
        """

        # first assume all layers will go onto gpu 0 as an optimizer. The
        # optimizer and embeddings are not quite as expensive as the
        # activations (which scale via batchsize), Empirically, I found this
        # heuristic works well enough. The weighting factor of 3 is more or
        # less made up.
        self.__device_allocations['cuda:0'] += trainable_parameters(model) * 3

        model.apply(self._place_modulelist)
        model._apply(self._move_rest_to_cuda0)  # type: ignore
        return model

    def _move_rest_to_cuda0(self, parameter: torch.Tensor):
        if parameter.device.type == 'cpu':
            return parameter.to('cuda:0')
        else:
            return parameter

    def _place_modulelist(self, submodule: torch.nn.Module) -> None:
        if not isinstance(submodule, torch.nn.ModuleList):
            # not a ModuleList, leave it untouched
            return
        if getattr(submodule, 'model_parallel_exempt', False):
            return

        assert isinstance(submodule, torch.nn.ModuleList)  # for typechecker
        layers = submodule

        # mark this section as MP
        layers.is_model_parallel = True  # type: ignore

        # next, let's figure out how many parameters we can assign to each GPU,
        # but not make actual assignments yet. Assignments come later because we
        # want consecutive layers to be collocated
        keyfunc = self.__device_allocations.__getitem__
        layer_assignments = {k: 0 for k in self.devices}
        for layer_no, layer in enumerate(layers):
            if layer_no == 0:
                # hard code the first layer to be 0.
                mostfree = 'cuda:0'
            else:
                # otherwise dynamic allocation
                mostfree = min(self.devices, key=keyfunc)
            # 32 is a totally arbitrary, made up number that worked in practice
            # on the large models I tested on. I believe it should be roughly
            # batch size, but this was set empirically.
            self.__device_allocations[mostfree] += trainable_parameters(layer) * 32
            # mark a layer as going to the given element
            layer_assignments[mostfree] += 1

        devices = [d for i, d in enumerate(self.devices[:]) if layer_assignments[d] > 0]
        for layer_no, layer in enumerate(layers):
            layer_gpu = devices[0]
            assert layer_assignments[layer_gpu] > 0
            logging.debug(f"Model Parallel: Assigning {layer_no} to {layer_gpu}")
            layer._mp_gpu = layer_gpu
            layers[layer_no] = layer.to(layer_gpu)
            layer_assignments[layer_gpu] -= 1
            if layer_assignments[layer_gpu] == 0:
                devices.pop(0)

    @staticmethod
    def guess_split_size(item: Chunk, num_gpus: Optional[int] = None, dim=0) -> int:
        """
        Estimate the number of chunks we should split the batch into via heuristics.
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()  # type: ignore

        if isinstance(item, torch.Tensor):
            if num_gpus == 1:
                # no point in chunking if we're not really doing model parallel
                return item.size(dim)
            # heuristic: use the same number of chunks as 2 * num_gpus.  this
            # isn't perfect (it ideally would be tuned differently for every model
            # and number of GPUs), but it seems to work reasonably wellenough in several
            # architectures tested.
            return max(1, item.size(dim) // int(num_gpus * 2))
        elif isinstance(item, tuple):
            return PipelineHelper.guess_split_size(item[0], num_gpus)
        elif isinstance(item, dict):
            return PipelineHelper.guess_split_size(list(item.values())[0], num_gpus)
        raise TypeError(f'Cannot determine split size for {type(item)}')

    @staticmethod
    def split(item: Chunk, split_size: Optional[int] = None, dim=0) -> List[Chunk]:
        """
        Split a tensor or group of tensors into smaller chunks of the same type.

        :param item:
            The item being split. May be a Tensor, a tuple of Tensors, or a
            dictionary mapping str -> Tensor.
        :param split_size:
            The maximum size of each output chunk. If None, we will guess using
            heuristics
        :param dim:
            The dimension to split along.
        """
        if split_size is None:
            split_size = PipelineHelper.guess_split_size(item)

        if isinstance(item, torch.Tensor):
            # base case, just split the tensor
            return list(torch.split(item, split_size, dim))
        elif isinstance(item, tuple):
            # We start with Tuple[Tensor] and we return List[Tuple[Tensor]]
            return list(zip(*(PipelineHelper.split(i, split_size, dim) for i in item)))
        elif isinstance(item, dict):
            if item == {}:
                # Terrible edge case: the empty dict. We handle by returning an
                # infinite list of empty dicts and we'll figure out its correct
                # size later. This happens for the incremental_state in
                # MultiheadAttention.
                return itertools.repeat({})  # type: ignore

            # we can't handle dicts with empty objects in them, due to how we handle
            # the case above.  awkward syntax because pytorch 1.3 doesn't like
            # comparing tensors to dicts.
            if {} in [x for x in item.values() if isinstance(x, dict)]:
                raise ValueError(
                    'Cannot handle a dictionary with an empty dictionary inside.'
                )
            if () in [x for x in item.values() if isinstance(x, tuple)]:
                raise ValueError(
                    'Cannot handle a dictionary with an empty tuple inside.'
                )

            # we start with Dict[key,tensor]
            # we map it to d: Dict[key, List[Tensor]], where we have split each mapping
            d = {k: PipelineHelper.split(v, split_size, dim) for k, v in item.items()}
            # now we transpose it and return List[Dict[key, Tensor]]
            return [
                dict(zip(d.keys(), values))  # type: ignore
                for values in zip(*(d[k] for k in d.keys()))
            ]
        else:
            raise TypeError(f"Cannot split type {type(item)}")

    @staticmethod
    def join(items: List[Chunk], dim=0) -> Chunk:
        """
        Join chunks back together, the inverse of split.

        :param items:
            All the output chunks. Each chunk may be a tensor or a group of
            tensors.
        :param dim:
            The dimension to join along.
        """
        if len(items) == 0:
            raise IndexError("Cannot rejoin an empty list of chunks.")
        item0 = items[0]
        if isinstance(item0, torch.Tensor):
            # base case
            return torch.cat(items, dim=dim)  # type: ignore
        elif isinstance(item0, tuple):
            return tuple(
                PipelineHelper.join(x, dim=dim) for x in zip(*items)
            )  # type: ignore
        elif isinstance(item0, dict):
            keys = item0.keys()
            return {  # type: ignore
                k: PipelineHelper.join([c[k] for c in items], dim=dim)  # type: ignore
                for k in keys
            }
        else:
            raise TypeError(f'Cannot join list of type {type(item0)}')

    @staticmethod
    def chunk_to(chunk: Chunk, device: str) -> Chunk:
        """
        Move the chunk to the device.

        Handles chunks which are groups of tensors.
        """
        if isinstance(chunk, torch.Tensor):
            return chunk.to(device)  # type: ignore
        elif isinstance(chunk, tuple):
            return tuple(
                PipelineHelper.chunk_to(c, device) for c in chunk
            )  # type: ignore
        elif isinstance(chunk, dict):
            return {
                k: PipelineHelper.chunk_to(v, device) for k, v in chunk.items()
            }  # type: ignore
        else:
            raise TypeError('chunk_to only compatible with tensors, tuples or dicts.')

    @staticmethod
    def schedule_work_items(layers: torch.nn.ModuleList, chunks: List[Chunk]):
        """
        Iterate through chunks and layers that should be pipelined.

        Each iteration of this generator yields the following properties:

            - layer_nos: a list of indices of layers for you to forward through
            - chunk_idx: the index of the chunk we are manipulating. Use this
              if you need to update chunk representations.
            - next_device: where the chunk should be moved to AFTER the layer
              computation is done.
        """
        # We want to pipeline our computations so that each GPU is working on
        # chunks of the problem at the same of the time. The load of the will
        # look like this, assuming there are 5 chunks (A, B, C, D, E) and 4
        # GPUs. Each slot fill means that gpu is working on that chunk.
        #
        #         +-----------------+
        #         |       Time      |
        #         | 1 2 3 4 5 6 7 8 |
        # +-------+-----------------+
        # |  G  0 | A B C D E       |
        # |  P  1 |   A B C D E     |
        # |  U  2 |     A B C D E   |
        # |     3 |       A B C D E |
        # +-------+-----------------+
        #
        # Note that some GPUs will be idle much of the time. In reality, we
        # will use 2 * num_gpus as the number of chunks, to minimize idle
        # time.
        num_chunks = len(chunks)
        for l in layers:
            if not hasattr(l, '_mp_gpu'):
                raise RuntimeError(
                    'You must run PipelineHelper.make_parallel on the ModuleList '
                    'before you can use iterate_layers_chunks.'
                )

        # devices maps device_idx -> (device, [layer_idx, layer_idx, ...])
        # for example, if devices is 2 and there are 4 layers, we might have
        #   devices = {
        #     0: ('cuda:0', [0]),
        #     1: ('cuda:1', [1, 2, 3]),
        #   }
        # This means layers 0 is on cuda:0, but layers 1-3 are on cuda:1.
        devices = {
            device_idx: (dev, list(grp))
            for device_idx, (dev, grp) in enumerate(
                itertools.groupby(range(len(layers)), lambda x: layers[x]._mp_gpu)
            )
        }
        num_timesteps = len(devices) + num_chunks
        for timestep in range(num_timesteps):
            for chunk_idx in range(num_chunks):
                device_idx = timestep - chunk_idx
                if device_idx >= 0 and device_idx < len(devices):
                    dev, layers_nos = devices[device_idx]
                    next_device, _ = devices[(device_idx + 1) % len(devices)]
                    assert device_idx in devices
                    yield PipelineWorkItem(
                        chunk_idx=chunk_idx,
                        layer_nos=layers_nos,
                        next_device=next_device,
                    )



class DynamicBatchWorld(World):
    def __init__(self, opt: Opt, world: Union[DialogPartnerWorld, MultiWorld]):
        super().__init__(opt)
        self.opt = opt

        # agents is a placeholder just for super.reset
        self.agents = []

        # check some assumptions
        if isinstance(world, (BatchWorld, MultiAgentDialogWorld)):
            raise TypeError(
                'World must be a DialogPartnerWorld or a '
                'MultiWorld of DialogPartnerWorld'
            )

        if len(world.get_agents()) != 2:
            raise AssertionError(
                "Dynamic batch only works in a fixed dialog world with two agents."
            )

        if not hasattr(world.get_model_agent(), 'batch_act'):
            raise TypeError("Model agent doesn't have batch_act.")

        self.truncate = opt.get('text_truncate', None) or opt.get('truncate', None)
        self.l_truncate = opt.get('label_truncate', None) or opt.get('truncate', None)
        if self.truncate is None or self.truncate < 0:
            raise ValueError(
                'You must use --text-truncate or --truncate in order to use '
                'dynamic batching.'
            )

        # size of the buffer we will use to find worlds
        if opt['dynamic_batching']:
            self._BUFFER_SIZE = 1021  # chosen as a prime number
        else:
            # we're secretly running in vanilla BS mode, via background
            # preprocessing
            self._BUFFER_SIZE = opt['batchsize']

        if opt['dynamic_batching'] == 'full':
            # full dynamic batching, we can grow our batchsize
            self.max_batch_size = self._BUFFER_SIZE
        else:
            # simple batchsort
            self.max_batch_size = opt['batchsize']

        # TODO: check to ensure the agent has self_observe
        self.world = world
        # TODO: maybe generalize this
        self.max_words = (self.l_truncate + self.truncate) * opt['batchsize']

        # buffer worlds
        self.worlds = [world.clone() for _ in range(self._BUFFER_SIZE)]

        self.reset()

    def shutdown(self):
        """
        Shutdown each world.
        """
        for w in self.worlds:
            w.shutdown()
        self.world.shutdown()

    def reset(self):
        super().reset()
        self._task_acts = [None for _ in range(self._BUFFER_SIZE)]
        self._obs = [None for _ in range(self._BUFFER_SIZE)]
        self._scores = [None for _ in range(self._BUFFER_SIZE)]
        self.acts = [None, None]

        self.number_parleys = 0
        self.total_exs = 0
        self.world.reset()
        self.rng = random.Random(4)
        for w in self.worlds:
            w.reset()

    def reset_metrics(self):
        super().reset_metrics()
        self.world.reset_metrics()
        for w in self.worlds:
            w.reset_metrics()

    def epoch_done(self):
        return (
            self.world.epoch_done()
            or all(w.epoch_done() for w in self.worlds)
            and all(s is None for s in self._scores)
        )

    def num_examples(self):
        return self.world.num_examples()

    def num_episodes(self):
        return self.world.num_episodes()

    def _ceil(self, n):
        """
        Round to the nearest multiple of 8.

        TensorCores only work when a tensor is a multiple of 8 in almost all
        dimensions. This means all examples cost is related to their nearest
        multiple of 8.

        See https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/ for
        more information.
        """
        # round up to r, all things are equal
        # from parlai.utils.torch import FP16_PAD_SIZE

        return ((n + FP16_PAD_SIZE - 1) // FP16_PAD_SIZE) * FP16_PAD_SIZE

    def _score(self, obs):
        if 'text_vec' in obs:
            # note that all examples have a cost that is based on their
            # nearest multiple of 4. We can therefore mix-and-match
            # anything with the same cost for increased stochasticity,
            # while not really wasting much padding.
            return tuple(
                self._ceil(len(obs[key]))
                for key in ['text_vec', 'labels_vec', 'eval_labels_vec']
                if key in obs
            )
        else:
            return None

    def parley(self):
        # first make sure that all the worlds are processed in the queue
        indices = []
        for i in range(self._BUFFER_SIZE):
            if self._scores[i] is not None:
                indices.append(i)
                continue
            if self.worlds[i].epoch_done():
                continue

            if hasattr(self.world, 'parley_init'):
                self.worlds[i].parley_init()

            act = self.worlds[i].get_task_agent().act()

            # we log the task act and the index of the act
            # in the buffer for world logging purposes
            self._task_acts[i] = act  # for world logging
            self._task_acts[i].force_set('dyn_batch_idx', i)

            obs = self.worlds[i].get_model_agent().observe(act)
            self._obs[i] = obs

            self._scores[i] = self._score(obs)
            if self._scores[i] is not None:
                indices.append(i)

        # quick invariant checks
        assert (
            len(indices) != 0 or self.world.num_examples() == 0
        ), "DynamicBatchWorld ran out of data!"
        assert not any(self._scores[i] is None for i in indices)

        if not indices:
            # this worker got no examples. This can happen when there are fewer
            # episodes than there are workers. "don't stress the small stuff."
            assert self.world.num_examples() == 0
            return

        # sort all the indices by their score, so that we can find similarly lengthed
        # items in O(1)
        indices = sorted(indices, key=lambda i: self._scores[i] + (self.rng.random(),))

        # now let's build the batch
        batch = []

        # start with a random item. indices_idx is the lookup into indices, but
        # index is the actual world.
        width = 0
        indices_idx = random.randint(0, len(indices) - 1)

        # we picked a random spot, but we can get better packing if we start at
        # the last example with the same score, since we always move down to
        # smaller examples.
        while indices_idx < len(indices) - 1 and (
            sum(self._scores[indices[indices_idx]])
            == sum(self._scores[indices[indices_idx + 1]])
        ):
            indices_idx += 1

        # quit early if we eat our full buffer
        while indices:
            index = indices[indices_idx]
            this_width = self._ceil(sum(self._scores[index]))
            new_width = max(width, this_width)
            # compute the cost of the new batch
            new_bsz = len(batch) + 1
            new_words = new_width * new_bsz
            if new_words <= self.max_words and new_bsz <= self.max_batch_size:
                # cool, this one fits, let's add it
                width = new_width
                batch.append(index)
                indices.pop(indices_idx)
                indices_idx = max(indices_idx - 1, 0)
            else:
                # we'd overfill our buffer, give up
                break

        # Always have a batch size that's a multiple of 4, for fp16's sake.
        while len(batch) > 4 and len(batch) % 4 != 0:
            # pop off the shortest one. it's easiest to pack in later
            batch.pop(-1)

        # double check our assumed invariant
        assert self._ceil(width) * len(batch) <= self.max_words
        assert len(batch) > 0
        assert len(batch) <= self.max_batch_size

        # great, this batch is good to go! let's run it!
        acts = self.handle_batch([self._obs[i] for i in batch])
        self.acts = [[self._task_acts[i] for i in batch], acts]
        # broadcast the results back to all the models
        for i, act in zip(batch, acts):
            # we need to make sure that the teachers saw the result
            self.worlds[i].get_task_agent().observe(act)
            # and that the agent copies saw their own voice
            self.worlds[i].get_model_agent().self_observe(act)
            # move these worlds forward
            act = self.worlds[i].get_task_agent().act()
            # we log the task act and the index of the act
            # in the buffer for world logging purposes
            self._task_acts[i] = act
            self._task_acts[i].force_set('dyn_batch_idx', i)
            # save the observations to form a batch
            obs = self.worlds[i].get_model_agent().observe(act)
            self._scores[i] = self._score(obs)
            self._obs[i] = obs

        # update metrics
        self.total_parleys += 1
        self.total_exs += len(batch)

    def handle_batch(self, batch):
        acts = self.world.get_model_agent().batch_act(batch)
        return acts

    def get_total_exs(self):
        return self.total_exs

    def get_total_epochs(self):
        return self.total_exs / self.num_examples()

    def report(self):
        return self.world.report()


class BackgroundDriverWorld(World):
    def __init__(self, opt: Opt, world: World):
        self.world = world
        super().__init__(opt, agents=world.agents, shared=None)

        import torch.multiprocessing as mp

        self._num_workers = self.opt['num_workers']
        # 4 per worker is somewhat arbitrary. 1 is potentially too few:
        # every worker is prevented from queuing up multiple batches.
        # Unbounded could fill up our memory too much. So 4 per worker.
        self._process_queue = mp.Queue(maxsize=4 * self._num_workers)
        self._process_pool = self._start_processes()

        self._batch_buffer = []
        self.metrics = TeacherMetrics()

    def _start_processes(self):
        import torch.multiprocessing as mp

        return mp.start_processes(
            fn=BackgroundWorkerDynamicBatchWorld.launch_process,
            nprocs=self._num_workers,
            # note that index is an an implied argument added by start_processes
            args=(self.opt, self.get_model_agent(), self._process_queue),
            join=False,
            # launch in fork mode so that we can share the model agent easily
            # note that this prevents us from using ANY threads in ANY of the
            # subprocesses! (See ChunkTeacher for one example). Fortunately, we
            # CAN use threads in the MAIN process, and we exploit this at
            # times.
            start_method='fork',
        )

    def reset(self):
        """
        Reset all subworlds.
        """
        self.world.reset()

    def reset_metrics(self):
        """
        Reset metrics in all subworlds.
        """
        self.world.reset_metrics()
        self.metrics.clear()

    def get_task_agent(self):
        return self.world.get_task_agent()

    def get_model_agent(self):
        return self.world.get_model_agent()

    def num_examples(self):
        return self.world.num_examples()

    def num_episodes(self):
        return self.world.num_episodes()

    def _queue_get(self):
        import queue

        while True:
            try:
                return self._process_queue.get(timeout=10)
            except queue.Empty:
                # not getting anything, let's check for exceptions on the
                self._process_pool.join(timeout=0.1)

    def parley(self):
        index, batch = self._queue_get()
        response_object = self.get_model_agent().batch_act(batch)
        # compute metrics
        for response in response_object:
            self.metrics._consume_user_metrics(response)
        self.total_parleys += 1
        self.total_exs += batch.batchsize

    def get_total_exs(self):
        return self.total_exs

    def get_total_epochs(self):
        return self.total_exs / self.num_examples()

    def report(self):
        return aggregate_unnamed_reports([self.world.report(), self.metrics.report()])

    def shutdown(self):
        logging.debug("Killing all the worker processes")
        for p in self._process_pool.processes:
            p.kill()
        super().shutdown()


class BackgroundWorkerDynamicBatchWorld(DynamicBatchWorld):
    @classmethod
    def launch_process(cls, index, opt, model_agent, process_queue):
        import torch

        torch.set_num_threads(1)  # prevent threads from spawning in this worker
        logging.info(f"Launching background on Index {index}")
        opt = copy.deepcopy(opt)
        opt['background_index'] = index
        try:
            world = cls(opt, model_agent=model_agent, process_queue=process_queue)
            while True:
                world.parley()
        except Exception:
            import traceback

            error = traceback.format_exc()
            logging.critical(
                f'Exception on background preprocesser index {index}!\n' + error
            )
            raise

    def __init__(self, opt: Opt, model_agent=None, process_queue=None):
        base_world = create_task_world(opt, [model_agent])
        self.process_queue = process_queue
        self.index = opt['background_index']
        super().__init__(opt, base_world)

    def handle_batch(self, batch):
        batchified = self.world.get_model_agent().batchify(batch)
        self.process_queue.put((self.index, batchified))
        acts = [{} for i in batch]
        return acts


################################################################################
# Functions for creating tasks/worlds given options.
################################################################################
def _create_task_agents(opt: Opt):
    """
    Create task agent(s) for the given task name.

    It does this by calling the create_agent function in agents.py of the given task. If
    create_agents function does not exist, it just looks for the teacher (agent) class
    defined by the task name directly.  (This saves the task creator bothering to define
    the create_agents function when it is not needed.)
    """
    if opt.get('interactive_task', False) or opt.get('selfchat_task', False):
        # do not need task agents in interactive or self chat settings
        return []

    try:
        # Tries to call the create_agent function in agents.py
        my_module = load_task_module(opt['task'])
        task_agents = my_module.create_agents(opt)  # type: ignore
    except (ModuleNotFoundError, AttributeError):
        # Create_agent not found, so try to create the teacher directly.
        return create_task_agent_from_taskname(opt)
    if type(task_agents) != list:
        task_agents = [task_agents]
    return task_agents


def create_task_world(opt: Opt, user_agents, default_world=None):
    """
    Instantiate a world with the supplied options and user agents.

    (A world factory.)
    """
    task_agents = _create_task_agents(opt)
    world_class = load_world_module(
        opt['task'],
        interactive_task=opt.get('interactive_task', False),
        selfchat_task=opt.get('selfchat_task', False),
        num_agents=len(user_agents + task_agents),
        default_world=default_world,
    )

    return world_class(opt, task_agents + user_agents)


def create_task(opt: Opt, user_agents, default_world=None):
    """
    Create a world + task_agents (aka a task).

    Assuming ``opt['task']="task_dir:teacher_class:options"`` e.g. ``"babi:Task1k:1"``
    or ``"#babi-1k"`` or ``"#QA"``, see ``parlai/tasks/tasks.py`` and see
    ``parlai/tasks/task_list.py`` for list of tasks.
    """
    task = opt.get('task')
    if not task:
        raise RuntimeError(
            'No task specified. Please select a task with ' + '--task {task_name}.'
        )
    if type(user_agents) != list:
        user_agents = [user_agents]

    # Convert any hashtag task labels to task directory path names.
    # (e.g. "#QA" to the list of tasks that are QA tasks).
    opt = copy.deepcopy(opt)
    opt['task'] = ids_to_tasks(opt['task'])
    logging.info(f"creating task(s): {opt['task']}")

    if ',' not in opt['task']:
        # Single task
        world = create_task_world(opt, user_agents, default_world=default_world)
    else:
        # Multitask teacher/agent
        # TODO: remove and replace with multiteachers only?
        world = MultiWorld(opt, user_agents, default_world=default_world)

    if DatatypeHelper.is_training(opt['datatype']) and opt.get('num_workers', 0) > 0:
        # note that we never use Background preprocessing in the valid/test
        # worlds, as we are unable to call Teacher.observe(model_act) in BG
        # preprocessing, so we are unable to compute Metrics or accurately
        # differentiate MultiWorld stats.
        world = BackgroundDriverWorld(opt, world)
    elif opt.get('batchsize', 1) > 1 and opt.get('dynamic_batching'):
        world = DynamicBatchWorld(opt, world)
    elif opt.get('batchsize', 1) > 1:
        # otherwise check if should use batchworld
        world = BatchWorld(opt, world)

    return world


def setup_interactive():
    """
    Set up the interactive script.
    """
    parser = setup_args()
    opt = parser.parse_args()
    # opt["model_file"] = 'D:/code/ParlAI-main/ParlAI/projects/image_chat/transresnet_multimodal/model/model'
    opt["model_file"] = "/home/nlp/CogAGENT/cogagent/toolkits/projects/image_chat/transresnet_multimodal/model/model"

    if not opt.get("model_file"):
        raise RuntimeError("Please specify a model file")
    if opt.get("fixed_cands_path") is None:
        fcp = os.path.join(
            "/".join(opt.get("model_file").split("/")[:-1]), "candidates.txt"
        )
        opt["fixed_cands_path"] = fcp
        opt["override"]["fixed_cands_path"] = fcp
    opt["task"] = "parlai.agents.local_human.local_human:LocalHumanAgent"
    opt["image_mode"] = "resnet152"
    opt["no_cuda"] = True
    opt["override"]["no_cuda"] = True
    SHARED["opt"] = opt
    SHARED["image_loader"] = ImageLoader(opt)

    # Create model and assign it to the specified task
    SHARED["agent"] = create_agent(opt, requireModelExists=True)
    SHARED["world"] = create_task(opt, SHARED["agent"])

    # Dialog History
    SHARED["dialog_history"] = []

    
import io
from base64 import b64decode


def interactive_running(data):
        """
        Generate a model response.

        :param data:
            data to send to model

        :return:
            model act dictionary
        """
        reply = {}
        if type(data["personality"][0]) is bytes:
            reply["text"] = data["personality"][0].decode("utf-8")
        else:
            reply["text"] = data["personality"][0]
        if type(data["text"][0]) is bytes:
            text = data["text"][0].decode("utf-8")
        else:
            text = data["text"][0]
        if text:
            reply["text"] = "\n".join(SHARED["dialog_history"] + [text, reply["text"]])
            SHARED["dialog_history"].append(text)
        if SHARED["image_feats"] is None:
            if type(data["image"][0]) is bytes:
                img_data = data["image"][0].decode("utf-8")
                _, encoded = img_data.split(",", 1)
                encoded = encoded[2:-1]
            else:
                img_data = data["image"][0]
                _, encoded = img_data.split(",", 1)
            image = Image.open(io.BytesIO(b64decode(encoded))).convert("RGB")
            # print("测试：", image)
            SHARED["image_feats"] = SHARED["image_loader"].extract(image)

        reply["image"] = SHARED["image_feats"]
        SHARED["agent"].observe(reply)
        model_res = SHARED["agent"].act()
        return model_res   


##############################################################
### WORLD LOADER
##############################################################
def _get_default_world(default_world=None, num_agents=None):
    """
    Get default world if a world is not already specified by the task.

    If a default world is provided, return this. Otherwise, return
    DialogPartnerWorld if there are 2 agents and MultiAgentDialogWorld if
    there are more.

    :param default_world:
        default world to return
    :param num_agents:
        number of agents in the environment
    """
    if default_world is not None:
        world_class = default_world
    elif num_agents is not None:
        # import parlai.core.worlds as core_worlds

        world_name = (
            "DialogPartnerWorld" if num_agents == 2 else "MultiAgentDialogWorld"
        )
        # print(os.path.basename(__file__).split('.py')[0])
        # print(globals()[world_name])
        # world_class = getattr(globals()[world_name], world_name)
        world_class = globals()[world_name]
    else:
        return None

    return world_class


import base64
def image_to_base64(path):
    with open(path, 'rb') as img:
        # 使用base64进行编码
        b64encode = base64.b64encode(img.read())
        s = b64encode.decode()
        b64_encode = 'data:image/jpeg;base64,%s' % s
        # 返回base64编码字符串
        return b64_encode


def inference(): # e_image, personality, text
    '''
    data = {
            'image':[e_image],
            'personality': [personality], 
            'text': [text]
        }
    if data["image"][0] != "":    # if e_image != "":
        SHARED["dialog_history"] = []
        SHARED["image_feats"] = None

    res = interactive_running(data=data)
    return res
    '''
    while True:
        text = input("我_文本:")
        e_image = input("我_图像:")
        personality = input("我_个性:")
        print("--" * 30)
        if e_image == '':
            e_image = ''
        else:
            e_image = image_to_base64(e_image)
        data = {
                    'image':[e_image],
                    'personality': [personality], 
                    'text': [text]
                }
        if data["image"][0] != "":    # if e_image != "":
            SHARED["dialog_history"] = []
            SHARED["image_feats"] = None
        
        res = interactive_running(data=data)
        print("问题:", text)
        print("答案:", res['text'])
        print("==" * 30)
        print()

        # return res

if __name__ == "__main__":
    setup_interactive()
    # img = Image.open("D:/code/ParlAI-main/ParlAI/my_project/dog.jpg") 
    # path = "D:/code/ParlAI-main/ParlAI/my_project/dog.jpg"
    # encode_image = image_to_base64(path)
    # print(encode_image[:100])
    '''
    # from ctypes import string_at
    # from sys import getsizeof
    # from binascii import hexlify
    # print(string_at(id(img),getsizeof(img)))
    
    # data = {
    #     'image':[encode_image],
    #     'personality': ['Adventurous'], 
    #     'text': ["What's in this picture?"]
    # }

    # if data["image"][0] != "":
    #     SHARED["dialog_history"] = []
    #     SHARED["image_feats"] = None

    # res = interactive_running(data=data)
    # print(res['text'])
    '''
    # while True:
    #     input_text = input("我_文本:")
    #     input_image = input("我_图像:")
    #     input_personality = input("我_个性:")
    #     print("--" * 30)
    #     if input_image == '':
    #         e_image = ''
    #     else:
    #         e_image = image_to_base64(input_image)
    #     res = inference(e_image, input_personality, input_text)
    #     print("问题:", input_text)
    #     print("答案:", res['text'])
    #     print("==" * 30)
    #     print()
    inference()