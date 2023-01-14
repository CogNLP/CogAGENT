import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from cogagent.models.base_model import BaseModel
from transformers import AutoModelForCausalLM,AutoConfig,AutoModelForMaskedLM

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def batch_cos(a, b, eps=1e-8):
    """
    https://stackoverflow.com/a/58144658
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


def tanh_clip(x, clip_val=None):
    if clip_val is not None:
        return clip_val * torch.tanh((1.0 / clip_val) * x)
    else:
        return x


def calc_regulaizer_coef(scores, regularizer_coef: float = 4e-2):
    return regularizer_coef * (scores ** 2.0).mean()



def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

class MaskRefineModel(BaseModel):
    def __init__(self,plm_name,mlm_name,vocab,
                 inbatch_negatives: bool = False,
                 stabilize: bool = True,
                 regularizer_coef: float = 4e-2,
                 clip_val: float = 10.0,
                 topK: int = 15,):
        super(MaskRefineModel, self).__init__()

        self.model_config =AutoConfig.from_pretrained(plm_name,return_dict=True,output_hidden_states=True)
        self.model = AutoModelForCausalLM.from_pretrained(plm_name,config=self.model_config,from_tf=False)
        self.model.resize_token_embeddings(new_num_tokens=vocab["new_num_tokens"])

        self.mlm_model_config = AutoConfig.from_pretrained(mlm_name,return_dict=True,output_hidden_states=True)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_name,config=self.mlm_model_config)

        self.ete = vocab["kge"].node_embds
        self.ete = nn.Embedding.from_pretrained(torch.Tensor(self.ete),freeze=False)
        self.kg_ent_head = nn.Linear(self.ete.embedding_dim, self.model.config.n_embd, bias=False)
        self.rte = vocab["kge"].rel_embds
        self.rte = nn.Embedding.from_pretrained(torch.Tensor(self.rte),freeze=False)
        self.kg_rel_head = nn.Linear(self.rte.embedding_dim, self.model.config.n_embd, bias=False)

        self.ln_transform = torch.nn.Linear(
            self.model.config.n_embd + self.mlm_model.config.hidden_size, self.model.config.n_embd, bias=False
        )
        self.init_ln_transform = torch.nn.Linear(
            self.mlm_model.config.hidden_size, self.model.config.n_embd, bias=False
        )

        self.inbatch_negatives = inbatch_negatives

        self.stabilize = stabilize
        self.regularizer_coef = regularizer_coef
        self.clip_val = clip_val

        self.topK = topK

    def loss(self, batch, loss_function):
        nce_batch,lm_batch = batch["nce_batch"],batch["lm_batch"]


        loss = 0
        if lm_batch is not None:
            lm_batch_selected = {k:v for k,v in lm_batch.items()
                                 if k not in ("triple_ids", "triple_type_ids")
                                 and (v is not None)
                                 }
            lm_batch_selected["labels"] = lm_batch_selected.pop("lm_labels")
            lm_output = self.model(**lm_batch_selected)
            ppl = torch.clamp(torch.exp(lm_output.loss), max=100, min=0)
            loss += lm_output.loss

        if nce_batch is not None:
            nce_loss = self.forward(nce_batch)
            loss += nce_loss

        return loss


    def forward(self,batch):
        mlm_inputs = batch["mlm_inputs"]
        lm_input_ids = batch["lm_input_ids"]
        lm_attention_masks = batch["lm_attention_masks"]
        candidate_ids = batch["candidate_ids"]
        candidate_rels = batch["candidate_rels"]
        pivot_ids = batch["pivot_ids"]
        pivot_fields = batch["pivot_fields"]
        labels = batch["labels"]
        label_indices = batch["label_indices"]
        mlm_entity_mask = batch["mlm_entity_mask"]
        inbatch_negatives = None

        if self.mlm_model is not None:
            mlm_output = self.mlm_model(**mlm_inputs)
            mlm_last_layer_hidden_states = mlm_output.hidden_states[-1]
            mlm_entity_reps = scatter_mean(mlm_last_layer_hidden_states, mlm_entity_mask, dim=1)[:, 1:, :]
        else:
            mlm_entity_reps = None

        B = lm_input_ids.shape[0]
        lm_range_tensor = torch.arange(B)

        prev_lm_entity_reps = None
        nb_masked_entities = lm_attention_masks.shape[1]

        nce_losses = []
        nce_accuracies = []
        all_logits = []
        selected_cands, topK_cands, topK_rels = [], [], []
        topK_pivots, topK_pivot_fields = [], []
        l_mrr, l_mr, l_hits1, l_hits3, l_hits10 = [], [], [], [], []

        for j in range(nb_masked_entities):
            lm_inputs_embeds = self.model.transformer.wte(lm_input_ids)

            for k in range(j + 1):
                entity_index = (lm_attention_masks[:, k, :] == 1).int().sum(-1) - 1
                if self.mlm_model is None:
                    entity_reps = self.lm_ete(mlm_inputs[:, k])
                    if j == k and prev_lm_entity_reps is not None:
                        entity_reps = self.ln_transform(torch.cat([prev_lm_entity_reps, entity_reps], dim=-1))
                    else:
                        entity_reps = self.kg_ent_head(entity_reps)
                else:
                    entity_reps = mlm_entity_reps[:, k, :]
                    if j == k and prev_lm_entity_reps is not None:
                        entity_reps = self.ln_transform(torch.cat([prev_lm_entity_reps, entity_reps], dim=-1))
                    else:
                        entity_reps = self.init_ln_transform(entity_reps)
                lm_inputs_embeds[lm_range_tensor, entity_index, :] = entity_reps

            # lm_output = B * hidden_state
            lm_output = self.model(inputs_embeds=lm_inputs_embeds)

            # lm_hidden_states = B * hidden_states
            lm_hidden_states = lm_output.hidden_states[-1][lm_range_tensor, entity_index]
            prev_lm_entity_reps = lm_hidden_states

            neighbor_embeds = self.kg_ent_head(self.ete(candidate_ids[:, j, ...]))
            pivot_embeds = self.kg_ent_head(self.ete(pivot_ids[:, j]))

            if len(pivot_embeds.shape) > 2:
                lm_hidden_states = lm_hidden_states.unsqueeze(1)

            object_embeds = pivot_embeds * lm_hidden_states
            H = object_embeds.shape[-1]
            N = neighbor_embeds.shape[2 if len(neighbor_embeds.shape) > 3 else 1]

            scores = torch.bmm(
                neighbor_embeds.view(-1, N, H), object_embeds.view(-1, H).unsqueeze(1).transpose(1, 2)
            ).squeeze(-1) / np.sqrt(H)

            scores = scores.reshape(B, -1)

            if inbatch_negatives or self.inbatch_negatives:
                inbatch_mask = (
                    (1 - torch.eye(B, device=scores.device))
                    .unsqueeze(-1)
                    .expand(B, B, neighbor_embeds.shape[1])
                    .reshape(B, -1)
                )
                mask = torch.cat([torch.ones_like(scores), inbatch_mask], dim=-1)

                inbatch_scores = torch.mm(
                    object_embeds, neighbor_embeds.view(-1, neighbor_embeds.shape[-1]).T
                ) / np.sqrt(H)
                scores = masked_log_softmax(torch.cat([scores, inbatch_scores], dim=-1), mask)
                current_candidate_ids = torch.cat(
                    [candidate_ids[:, j], candidate_ids[:, j].reshape(1, -1).repeat(B, 1)], dim=-1
                )
                current_candidate_rels = torch.cat(
                    [candidate_rels[:, j], candidate_rels[:, j].reshape(1, -1).repeat(B, 1)], dim=-1
                )
            else:
                scores = F.log_softmax(scores, dim=-1)
                current_candidate_ids = candidate_ids[:, j].reshape(B, -1)
                current_candidate_rels = candidate_rels[:, j].reshape(B, -1)

            if self.stabilize:
                scores = tanh_clip(scores, self.clip_val)
                reg = calc_regulaizer_coef(scores, self.regularizer_coef)
            else:
                reg = 0.0

            _, max_score_indices = torch.max(scores, dim=1)
            selected_cands.append(current_candidate_ids[torch.arange(B), max_score_indices].unsqueeze(0))

            topK_scores, topK_score_indices = torch.topk(scores, k=min(self.topK, scores.shape[1]), dim=1, largest=True)
            topK_cands.append(torch.gather(current_candidate_ids, dim=-1, index=topK_score_indices).unsqueeze(0))
            topK_rels.append(torch.gather(current_candidate_rels, dim=-1, index=topK_score_indices).unsqueeze(0))

            if pivot_fields is not None:
                topK_pivot_fields.append(
                    torch.gather(pivot_fields[:, j], dim=-1, index=topK_score_indices // N).unsqueeze(0)
                )

                topK_pivots.append(
                    torch.gather(pivot_ids[:, j], dim=-1, index=topK_score_indices // N).unsqueeze(0)
                )

            if label_indices is not None:
                current_labels = label_indices[:, j]
                accuracy = (max_score_indices == current_labels).float().sum() / (current_labels >= 0).int().sum()

                nonpad_labels = torch.masked_select(current_labels, current_labels.ge(0))
                ranking_metrics = self.compute_ranks(scores[current_labels >= 0, :], nonpad_labels)

                l_mr.append(torch.FloatTensor([ranking_metrics["MR"]]))
                l_mrr.append(torch.FloatTensor([ranking_metrics["MRR"]]))
                l_hits1.append(torch.FloatTensor([ranking_metrics["HITS@1"]]))
                l_hits3.append(torch.FloatTensor([ranking_metrics["HITS@3"]]))
                l_hits10.append(torch.FloatTensor([ranking_metrics["HITS@10"]]))
                nce_accuracies.append(accuracy)

                pos_scores = scores[current_labels >= 0, nonpad_labels]
                nce_loss = -pos_scores.mean() + reg
                nce_losses.append(nce_loss)

            all_logits.append(scores)

        return torch.mean(torch.vstack(nce_losses)) if nce_losses is not None else None
        # return MaskRefineOutput(
        #     all_logits,
        #     torch.vstack(selected_cands).T,
        #     torch.vstack(topK_cands).transpose(1, 0),
        #     torch.vstack(topK_rels).transpose(1, 0),
        #     torch.vstack(topK_pivots).transpose(1, 0) if topK_pivots else None,
        #     torch.vstack(topK_pivot_fields).transpose(1, 0) if topK_pivot_fields else None,
        #     torch.vstack(nce_losses) if nce_losses else None,
        #     torch.vstack(nce_accuracies) if nce_accuracies else None,
        #     torch.vstack(l_mr) if l_mr else None,
        #     torch.vstack(l_mrr) if l_mrr else None,
        #     torch.vstack(l_hits1) if l_hits1 else None,
        #     torch.vstack(l_hits3) if l_hits3 else None,
        #     torch.vstack(l_hits10) if l_hits10 else None,
        # )

    def compute_ranks(self, scores, current_labels):
        argsort = torch.argsort(scores, dim=1, descending=True)
        batch_size = scores.shape[0]
        logs = []
        for i in range(batch_size):
            # Notice that argsort is not rankingc
            # Returns the rank of the current_labels in argsort
            ranking = torch.nonzero(argsort[i, :] == current_labels[i], as_tuple=False)
            assert ranking.shape[0] == 1

            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            logs.append(
                {
                    "MRR": 1.0 / ranking,
                    "MR": float(ranking),
                    "HITS@1": 1.0 if ranking <= 1 else 0.0,
                    "HITS@3": 1.0 if ranking <= 3 else 0.0,
                    "HITS@10": 1.0 if ranking <= 10 else 0.0,
                }
            )
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics



if __name__ == '__main__':
    from cogagent.data.readers.open_dialkg_reader import OpenDialKGReader
    from cogagent.data.processors.open_dialkg_processors.open_dialkg_for_nph_processor import OpenDialKGForNPHProcessor
    from cogagent.utils.log_utils import init_logger
    logger = init_logger()
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data",debug=True)
    # train_data,dev_data,test_data = reader.read_all()
    train_data = reader._read_train()
    vocab = reader.read_vocab()

    plm_name = 'gpt2'
    mlm_name = 'roberta-large'

    processor = OpenDialKGForNPHProcessor(vocab=vocab,plm=plm_name,mlm=mlm_name,debug=True)
    train_dataset = processor.process_train(train_data)

    # test model construction
    model = MaskRefineModel(plm_name=plm_name,mlm_name=mlm_name,vocab=vocab)

    # test model forward
    device = torch.device("cuda:2")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, collate_fn=processor._collate)
    batch = next(iter(dataloader))

    from cogagent.utils.train_utils import move_dict_value_to_device
    move_dict_value_to_device(batch ,device=device)
    model.to(device)
    loss = model.loss(batch,None)
    print("loss=",loss.item())
    print("Finished!")



