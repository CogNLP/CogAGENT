import json
import torch
import pickle
import torch.nn as nn
from pathlib import Path
import yaml
import os


def load_json(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as f:
        json.dump(data, f, indent=4)


def load_pickle(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'wb') as f:
        pickle.dump(data, f)


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    print(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states, strict=False)
    return model


def save_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_yaml(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    config = None
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_file_path_yaml(file_path):
    config = load_yaml(file_path)
    if config is not None:
        for key, value in config["openqa"].items():
            path = os.path.join(config["rootpath"], value)
            if os.path.exists(path):
                config["openqa"][key] = path
        for key, value in config["dialogue_safety"].items():
            path = os.path.join(config["rootpath"], value)
            if os.path.exists(path):
                config["dialogue_safety"][key] = path
        for key, value in config["sticker"].items():
            path = os.path.join(config["rootpath"], value)
            if os.path.exists(path):
                config["sticker"][key] = path
    return config
