import torch
import collections
from torch.utils.data import Dataset
from cogagent.data.datable import DataTable


class DataTableSet(Dataset):
    def __init__(self, datable: DataTable, name=None, to_device=True, shuffle=True):
        self.datable = datable
        self.length = len(self.datable)
        self.type2type = {}
        self.type2type[int] = torch.long
        self.type2type[float] = torch.float
        self.type2type[bool] = torch.bool
        self._name = name
        self.shuffle = shuffle

    def __getitem__(self, index):
        item = []
        for header in self.datable.headers:
            if header not in self.datable.not2torch:
                data = self.datable.datas[header][index]
                while True:
                    if isinstance(data, list) or isinstance(data, tuple):
                        if len(data) == 0:
                            item.append(self.datable.datas[header][index])
                            break
                        else:
                            data = data[0]
                    else:
                        if type(data) in self.type2type:
                            dtype = self.type2type[type(data)]
                            item.append(
                                torch.tensor(self.datable.datas[header][index], dtype=dtype))
                        else:
                            item.append(self.datable.datas[header][index])
                        break
            else:
                item.append(self.datable.datas[header][index])
        return {key: value for key, value in zip(self.datable.headers, item)}

    def __len__(self):
        return self.length

    def to_dict(self, batch):
        batch = [list(item.values()) for item in batch]
        batch = list(map(list, zip(*batch)))
        data = collections.OrderedDict()
        for i in range(len(batch)):
            if isinstance(batch[i][0], torch.Tensor):
                data[self.datable.headers[i]] = torch.stack(batch[i])
            else:
                data[self.datable.headers[i]] = batch[i]
        return data