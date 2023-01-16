from cogagent.toolkits.base_toolkit import BaseToolkit
from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification
from cogagent.utils.constant.opendialkg_constants import SPECIAL_TOKENS,ATTR_TO_SPECIAL_TOKEN,SpecialTokens

import logging

from typing import Dict, Iterable, Optional, List

import torch
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering
from transformers.file_utils import ModelOutput

class KGGroundedConversationAgent(BaseToolkit):
    def __init__(self):
        super(KGGroundedConversationAgent, self).__init__()

        # load hallucination classifier
        self.halluc_classifier = "/data/hongbang/projects/Neural-Path-Hunter/data/best_model"
        self.halluc_config = AutoConfig.from_pretrained(self.halluc_classifier,return_dict=True,output_hidden_states=True)
        self.halluc_tokenizer = AutoTokenizer.from_pretrained(self.halluc_classifier)
        self.halluc_classifier = AutoModelForTokenClassification.from_pretrained(self.halluc_classifier)
        self.halluc_special_tokens = SpecialTokens(*self.halluc_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))





logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def generate_no_beam_search(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Dict[str, torch.LongTensor]:
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """
    config: PretrainedConfig = getattr(model, "module", model).config

    max_length = max_length if max_length is not None else config.max_length
    min_length = min_length if min_length is not None else config.min_length
    do_sample = do_sample if do_sample is not None else config.do_sample
    use_cache = use_cache if use_cache is not None else config.use_cache
    temperature = temperature if temperature is not None else config.temperature
    top_k = top_k if top_k is not None else config.top_k
    top_p = top_p if top_p is not None else config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
    no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else config.no_repeat_ngram_size
    bad_words_ids = bad_words_ids if bad_words_ids is not None else config.bad_words_ids
    num_return_sequences = num_return_sequences if num_return_sequences is not None else config.num_return_sequences
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else config.decoder_start_token_id
    )

    batch_size = input_ids.shape[0]

    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if pad_token_id is None and eos_token_id is not None:
        logger.warning("Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id))
        pad_token_id = eos_token_id

    # set effective batch size and effective batch multiplier according to do_sample

    effective_batch_size = batch_size * num_return_sequences
    effective_batch_mult = num_return_sequences

    if config.is_encoder_decoder:
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif (
                hasattr(config, "decoder")
                and hasattr(config.decoder, "bos_token_id")
                and config.decoder.bos_token_id is not None
            ):
                decoder_start_token_id = config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()
        encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)

    # Expand input ids if num_return_sequences > 1
    if num_return_sequences > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, effective_batch_mult, input_ids_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).expand(-1, effective_batch_mult, input_ids_len)
            token_type_ids = token_type_ids.contiguous().view(effective_batch_size, input_ids_len)

        input_ids = input_ids.contiguous().view(
            effective_batch_size, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if config.is_encoder_decoder:
        device = next(model.parameters()).device
        if decoder_input_ids is not None:
            # give initial decoder input ids
            input_ids = decoder_input_ids.repeat(effective_batch_size, 1).to(device)
        else:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

        assert (
            batch_size == encoder_outputs.last_hidden_state.shape[0]
        ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size).view(-1, 1).repeat(1, effective_batch_mult).view(-1).to(input_ids.device)
        )

        # expand encoder_outputs
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)

        # save encoder_outputs in `model_kwargs`
        model_kwargs["encoder_outputs"] = encoder_outputs

    input_lengths = (input_ids != pad_token_id).int().sum(-1) - 1

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(effective_batch_size).fill_(1)
    sent_lengths = input_ids.new(effective_batch_size).fill_(max_length)

    generated_ids = input_ids[torch.arange(effective_batch_size), input_lengths].unsqueeze(-1)

    if token_type_ids is not None:
        generated_token_types = token_type_ids[torch.arange(effective_batch_size), input_lengths].unsqueeze(-1)

    past = None
    for cur_len in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )

        if token_type_ids is not None:
            if past:
                model_inputs["token_type_ids"] = token_type_ids[:, -1].unsqueeze(-1)
            else:
                model_inputs["token_type_ids"] = token_type_ids

        outputs = model(**model_inputs, return_dict=True)
        if cur_len == 0:
            next_token_logits = outputs.logits[torch.arange(effective_batch_size), input_lengths, :]
        else:
            next_token_logits = outputs.logits[:, -1, :]

        scores = postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Greedy decoding
            if cur_len == 0:
                _, next_token = torch.topk(
                    next_token_logits.view(batch_size, effective_batch_mult, -1)[:, 0, :],
                    k=effective_batch_mult,
                    dim=-1,
                )
                next_token = next_token.reshape(effective_batch_size, -1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        if token_type_ids is not None:
            next_token_types = torch.gather(token_type_ids, dim=1, index=input_lengths.unsqueeze(-1)).squeeze(
                -1
            ) * unfinished_sents + pad_token_id * (1 - unfinished_sents)
        next_len = cur_len + 1
        input_lengths = input_lengths + 1
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        generated_ids = torch.cat([generated_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if token_type_ids is not None:
            token_type_ids = torch.cat([token_type_ids, next_token_types.unsqueeze(-1)], dim=-1)
            generated_token_types = torch.cat([generated_token_types, next_token_types.unsqueeze(-1)], dim=-1)

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, next_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    output = dict(input_ids=input_ids, generated_ids=generated_ids, attention_mask=attention_mask)
    if token_type_ids is not None:
        output["token_type_ids"] = token_type_ids
        output["generated_token_types"] = generated_token_types

    return output


def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
):

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # Exclude EOS token (already processed)
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
        # Modify the scores in place by setting the banned tokens logits to `-inf`
        set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    return scores



def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))


if __name__ == '__main__':
    print("Hello World!")
    agent = KGGroundedConversationAgent()
    # agent = KGGroundedConversationAgent(
    #     bert_model=None,
    #     model_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result/run_diffks_wow_lr1e-4--2022-11-16--01-01-39.05/best_model/checkpoint-376000/models.pt',
    #     vocabulary_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/wow_vocab.pkl',
    #     device=torch.device("cuda:0"),
    #     debug=False,
    #     )
    # agent.run()

    # topic = agent.choose_topic()
    # print(topic)
    # sentence = "Downald trump is not doing well recently."
    # label = toolkit.run(sentence)
    # print("label:",label)


