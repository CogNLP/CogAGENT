from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from tqdm import tqdm
import transformers
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from cogagent.data.processors.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()


class WayProcessor(BaseProcessor):

    def __init__(self, addition, do_sample=False):
        super().__init__()
        self.addition_dict = addition
        self.mesh2meters = addition['mesh2meters']
        self.floorplans_path = addition['floorplans_path']
        self.do_sample = do_sample
        self.ds_percent = 0.65
        self.max_floors = 5
        self.image_size = [
            3,
            int(700 * self.ds_percent),
            int(1200 * self.ds_percent),
        ]

        self.preprocess_data_aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.555],
                    std=[0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.555],
                    std=[0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

    def gather_all_floors(self, index, data, mode):
        all_maps = torch.zeros(
            self.max_floors,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        all_conversions = torch.zeros(self.max_floors, 1)
        sn = data["scan_names"][index]
        floors = self.mesh2meters[sn].keys()
        img_list = []
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.floorplans_path, f, sn, f)
            )
            img = img.resize((self.image_size[2], self.image_size[1]))
            img_list.append(img)
            if "train" in mode:
                all_maps[enum, :, :, :] = self.preprocess_data_aug(img)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
            all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
        return all_maps, all_conversions, img_list

    def create_target(self, index, location, mesh_conversion, data):
        gaussian_target = np.zeros(
            (self.max_floors, self.image_size[1], self.image_size[2])
        )
        gaussian_target[int(data["levels"][index]), location[0], location[1]] = 1
        gaussian_target[int(data["levels"][index]), :, :] = gaussian_filter(
            gaussian_target[int(data["levels"][index]), :, :],
            sigma=mesh_conversion,
        )
        gaussian_target[int(data["levels"][index]), :, :] = (
                gaussian_target[int(data["levels"][index]), :, :]
                / gaussian_target[int(data["levels"][index]), :, :].sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        return gaussian_target

    def get_info(self, index, data):
        info_elem = [
            data["dialogs"][index],
            data["levels"][index],
            data["scan_names"][index],
            data["episode_ids"][index],
            data["viewPoint_location"][index],
        ]
        return info_elem

    def _process(self, data, mode=None):
        if self.do_sample:
            data = self._do_sample_process(data=data, do_sample=self.do_sample)
        datable = DataTable()
        print("Processing data...")
        for index, (texts, \
                    seq_lengths, \
                    mesh_conversions, \
                    locations, \
                    viewPoint_location, \
                    dialogs, \
                    scan_names, \
                    levels, \
                    episode_ids) in enumerate(tqdm(zip(data['texts'],
                                                       data['seq_lengths'],
                                                       data['mesh_conversions'],
                                                       data['locations'],
                                                       data['viewPoint_location'],
                                                       data['dialogs'],
                                                       data['scan_names'],
                                                       data['levels'],
                                                       data['episode_ids']),
                                                   total=len(data['texts']))):
            location = np.round(np.asarray(locations) * self.ds_percent).astype(int)
            mesh_conversion = mesh_conversions * self.ds_percent
            text = torch.LongTensor(texts)
            seq_length = torch.tensor(seq_lengths)
            maps, conversions, img_list = self.gather_all_floors(index=index, data=data, mode=mode)
            target = self.create_target(index, location, mesh_conversion, data)
            info_elem = self.get_info(index, data)

            datable("info_elem", info_elem)
            datable("text", text)
            datable("seq_length", seq_length)
            datable("target", target)
            datable("maps", maps)
            datable("conversions", conversions)
            datable("img_list", img_list)

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data=data, mode="train")

    def process_dev_seen(self, data):
        return self._process(data=data, mode="valSeen")

    def process_dev_unseen(self, data):
        return self._process(data=data, mode="valUnseen")

    def process_test(self, data):
        return self._process(data=data, mode="test")

    def get_addition(self):
        return self.addition_dict


if __name__ == "__main__":
    from cogagent.data.readers.way_reader import WayReader

    reader = WayReader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/embodied_dialog/way/raw_data")
    train_data, dev_seen_data, dev_unseen_data, test_data = reader.read_all()
    addition = reader.read_addition()

    processor = WayProcessor(
        addition=addition,
        do_sample=True)
    train_dataset = processor.process_train(train_data)
    dev_seen_dataset = processor.process_dev_seen(dev_seen_data)
    dev_unseen_dataset = processor.process_dev_unseen(dev_unseen_data)
    test_dataset = processor.process_test(test_data)
    addition = processor.get_addition()
    print("end")
