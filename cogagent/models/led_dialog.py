from cogagent.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torchvision.models as models
import json
import math
import matplotlib.pyplot as plt


def accuracy(dists, threshold=3):
    return np.mean((torch.tensor(dists) <= threshold).int().numpy())


def snap_to_grid(geodistance_nodes, node2pix, sn, pred_coord, conversion, level):
    min_dist = math.inf
    best_node = ""
    for node in node2pix[sn].keys():
        if node2pix[sn][node][2] != int(level) or node not in geodistance_nodes:
            continue
        target_coord = [node2pix[sn][node][0][1], node2pix[sn][node][0][0]]
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        if dist.item() < min_dist:
            best_node = node
            min_dist = dist.item()
    return best_node


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LinearProjectionLayers(nn.Module):
    def __init__(
            self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers
    ):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            self.out_layers = nn.Sequential(
                nn.Conv2d(
                    image_channels + rnn_hidden_size,
                    linear_hidden_size,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.ReLU(),
                nn.Conv2d(linear_hidden_size, 1, kernel_size=1, padding=0, stride=1),
            )
            self.out_layers.apply(self.init_weights)

    def forward(self, x):
        return self.out_layers(x)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class RNN(nn.Module):
    def __init__(
            self,
            input_size,
            embed_size,
            hidden_size,
            num_layers,
            dropout,
            bidirectional,
            embedding_dir,
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reduce = "last" if not bidirectional else "mean"
        self.embedding_dir = embedding_dir

        self.embedding = nn.Embedding(input_size, embed_size)
        glove_weights = torch.FloatTensor(
            np.load(self.embedding_dir + "glove_weights_matrix.npy", allow_pickle=True)
        )
        self.embedding.from_pretrained(glove_weights)

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        embed_packed = pack_padded_sequence(
            embed, seq_lengths.cpu(), enforce_sorted=False, batch_first=True
        )

        out_packed = embed_packed
        self.lstm.flatten_parameters()
        out_packed, _ = self.lstm(out_packed)
        out, _ = pad_packed_sequence(out_packed)

        # reduce the dimension
        if self.reduce == "last":
            out = out[seq_lengths - 1, np.arange(len(seq_lengths)), :]
        elif self.reduce == "mean":
            seq_lengths_ = seq_lengths.unsqueeze(-1)
            out = torch.sum(out[:, np.arange(len(seq_lengths_)), :], 0) / seq_lengths_

        return out


class LedDialog(BaseModel):

    def __init__(self, addition, embedding_dir):
        super().__init__()
        self.addition = addition
        self.node2pix_path = addition["node2pix_path"]
        self.geodistance_nodes_path = addition["geodistance_nodes_path"]
        self.node2pix = json.load(open(self.node2pix_path))
        self.geodistance_nodes = json.load(open(self.geodistance_nodes_path))
        self.embedding_dir = embedding_dir

        self.m = 3  # num_lingunet_layers
        self.image_channels = 128  # linear_hidden_size
        self.freeze_resnet = True
        self.res_connect = True
        self.bidirectional = True
        self.rnn_hidden_size = 300
        self.input_size = len(self.addition["vocab"])
        self.embed_size = 300
        self.num_rnn_layers = 1
        self.embed_dropout = 0.5
        self.num_linear_hidden_layers = 1
        self.ds_height = 57
        self.ds_width = 98
        self.max_floors = 5

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        if self.freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        if not self.bidirectional:
            self.rnn_hidden_size = self.rnn_hidden_size
        else:
            self.rnn_hidden_size = self.rnn_hidden_size * 2
        assert self.rnn_hidden_size % self.m == 0

        self.rnn = RNN(
            self.input_size,
            self.embed_size,
            self.rnn_hidden_size,
            self.num_rnn_layers,
            self.embed_dropout,
            self.bidirectional,
            self.embedding_dir,
        )

        sliced_text_vector_size = self.rnn_hidden_size // self.m
        flattened_conv_filter_size = 1 * 1 * self.image_channels * self.image_channels
        self.text2convs = clones(
            nn.Linear(sliced_text_vector_size, flattened_conv_filter_size), self.m
        )

        self.conv_layers = nn.ModuleList([])
        for i in range(self.m):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.image_channels
                        if i == 0
                        else self.image_channels,
                        out_channels=self.image_channels,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                    ),
                    nn.BatchNorm2d(self.image_channels),
                    nn.ReLU(True),
                )
            )

        # create deconv layers with appropriate paddings
        self.deconv_layers = nn.ModuleList([])
        for i in range(self.m):
            in_channels = self.image_channels if i == 0 else self.image_channels * 2
            out_channels = self.image_channels
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

        self.conv_dropout = nn.Dropout(p=0.25)
        self.deconv_dropout = nn.Dropout(p=0.25)

        self.out_layers = LinearProjectionLayers(
            image_channels=self.image_channels,
            linear_hidden_size=self.image_channels,
            rnn_hidden_size=0,
            num_hidden_layers=self.num_linear_hidden_layers,
        )
        self.sliced_size = self.rnn_hidden_size // self.m

        # initialize weights
        self.text2convs.apply(self.init_weights)
        self.conv_layers.apply(self.init_weights)
        self.deconv_layers.apply(self.init_weights)
        self.out_layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)

        B, num_maps, C, H, W = batch['maps'].size()
        batch_target = batch['target'].view(B * num_maps, H, W)

        batch_target = (
            nn.functional.interpolate(
                batch_target.unsqueeze(1),
                (self.ds_height, self.ds_width),
                mode="bilinear",
            )
                .squeeze(1)
                .float()
        )
        batch_target = batch_target.view(
            B, num_maps, batch_target.size()[-2], batch_target.size()[-1]
        )

        loss = loss_function(pred, batch_target)
        return loss

    def forward(self, batch):
        text = batch['text']
        seq_length = batch['seq_length']
        maps = batch['maps']

        images = maps
        texts = text
        seq_lengths = seq_length

        B, num_maps, C, H, W = images.size()
        images = images.view(B * num_maps, C, H, W)
        images = self.resnet(images)

        batch_size, image_channels, height, width = images.size()

        text_embed = self.rnn(texts, seq_lengths)
        text_embed = torch.repeat_interleave(text_embed, num_maps, dim=0)

        Gs = []
        image_embeds = images

        for i in range(self.m):
            image_embeds = self.conv_dropout(image_embeds)
            image_embeds = self.conv_layers[i](image_embeds)
            text_slice = text_embed[
                         :, i * self.sliced_size: (i + 1) * self.sliced_size
                         ]

            conv_kernel_shape = (
                batch_size,
                self.image_channels,
                self.image_channels,
                1,
                1,
            )
            text_conv_filters = self.text2convs[i](text_slice).view(conv_kernel_shape)

            orig_size = image_embeds.size()
            image_embeds = image_embeds.view(1, -1, *image_embeds.size()[2:])
            text_conv_filters = text_conv_filters.view(
                -1, *text_conv_filters.size()[2:]
            )
            G = F.conv2d(image_embeds, text_conv_filters, groups=orig_size[0]).view(
                orig_size
            )
            image_embeds = image_embeds.view(orig_size)
            if self.res_connect:
                G = G + image_embeds
                G = F.relu(G)
            Gs.append(G)

        H = Gs.pop()
        for i in range(self.m):
            if i == 0:
                H = self.deconv_dropout(H)
                H = self.deconv_layers[i](H)
            else:
                G = Gs.pop()
                concated = torch.cat((H, G), 1)
                H = self.deconv_layers[i](concated)

        out = self.out_layers(H).squeeze(-1)
        out = out.view(B, num_maps, out.size()[-2], out.size()[-1])
        out = F.log_softmax(out.view(B, -1), 1).view(B, num_maps, height, width)
        return out

    def evaluate(self, batch, metric_function):
        preds, result_list = self.predict(batch)

        info_elem_old = batch['info_elem']
        conversions = batch['conversions']
        device = preds.device

        info_elem = [[], [], [], [], []]
        for item in info_elem_old:
            info_elem[0].append(item[0])
            info_elem[1].append(item[1])
            info_elem[2].append(item[2])
            info_elem[3].append(item[3])
            info_elem[4].append(item[4])

        distances = []
        dialogs, levels, scan_names, episode_ids, true_viewpoints = info_elem
        for pred, conversion, sn, tv, id in zip(
                preds, conversions, scan_names, true_viewpoints, episode_ids
        ):
            total_floors = len(set([v[2] for k, v in self.node2pix[sn].items()]))
            pred = nn.functional.interpolate(
                pred.unsqueeze(1), (700, 1200), mode="bilinear"
            ).squeeze(1)[:total_floors].cpu()  #
            pred_coord = np.unravel_index(pred.argmax(), pred.size())
            convers = conversion.view(self.max_floors, 1, 1)[pred_coord[0].item()]
            pred_viewpoint = snap_to_grid(
                self.geodistance_nodes[sn],
                self.node2pix,
                sn,
                [pred_coord[1].item(), pred_coord[2].item()],
                convers,
                pred_coord[0].item(),
            )
            dist = self.geodistance_nodes[sn][tv][pred_viewpoint]
            distances.append(dist)

        metric_function.evaluate(pred=distances)

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            preds = self.forward(batch)

            batch_size = len(preds)
            B, num_maps, C, H, W = batch['maps'].size()
            H_factor = H / self.ds_height
            W_factor = W / self.ds_width

            # #################
            # batch_target = batch['target'].view(B * num_maps, H, W)
            #
            # batch_target = (
            #     nn.functional.interpolate(
            #         batch_target.unsqueeze(1),
            #         (self.ds_height, self.ds_width),
            #         mode="bilinear",
            #     )
            #         .squeeze(1)
            #         .float()
            # )
            # batch_target = batch_target.view(
            #     B, num_maps, batch_target.size()[-2], batch_target.size()[-1]
            # )
            # #################

            result_list = []
            for i in range(batch_size):
                result = np.unravel_index(preds[i].detach().cpu().argmax(), preds[i].size())
                floor, w_loc, h_loc = result
                h_loc = int(H_factor * h_loc)
                w_loc = int(W_factor * w_loc)
                plt.imshow(batch['img_list'][i][floor])
                plt.plot(h_loc, w_loc, "ro")
                plt.show()

                ##########################
                # label=np.unravel_index(batch_target[i].detach().cpu().argmax(), batch_target[i].size())
                # label_floor, label_w_loc, label_h_loc = label
                # label_h_loc = int(H_factor * label_h_loc)
                # label_w_loc = int(W_factor * label_w_loc)
                # plt.imshow(batch['img_list'][i][label_floor])
                # plt.plot(label_h_loc,label_w_loc, "ro",color='lime')
                # plt.show()
                ##########################

                result_dict = {}
                result_dict["h_loc"] = h_loc
                result_dict["w_loc"] = w_loc
                result_dict["floor"] = floor
                result_dict["img"] = batch['img_list'][i][floor]
                result_list.append(result_dict)

        return preds, result_list
