import json
import torch
import pickle
import torch.nn as nn
from pathlib import Path


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