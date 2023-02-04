import argparse
import json
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import wordnet
from cogagent.data.processors.kemp_processors.kemp_dataloader import Dataset
from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
import torch
import numpy as np
from collections import defaultdict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
transformers.logging.set_verbosity_error()  # set transformers logging level
EOS_token = 1
PAD_token = 3
class KempProcessor(BaseProcessor):
    def __init__(self, word2index, args):
        super().__init__()
        self.word2index = word2index
        self.args = args
    
    def _process(self, data):
        dataset_train = Dataset(data, self.word2index, self.args)
        return dataset_train

    def process_train(self, data):
        print("Processing train data...")
        return self._process(data)

    def process_dev(self, data):
        print("Processing val data...")
        return self._process(data)

    def process_test(self, data):
        print("Processing test data...")
        return self._process(data)
        
def preprocess(sentence, concept_num=3):
        # with open('/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data/dataset_preproc.json', "r") as f:
        #     [data_tra, data_val, data_sent, vocab] = json.load(f)
        #     word2index, word2count, index2word, n_words = vocab
        with open('/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data/vocab.json', "r") as f:
            vocab = json.load(f)
            word2index, word2count, index2word, n_words = vocab
        data_sent = {}
        VAD = json.load(open("/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data/VAD.json", "r", encoding="utf-8"))
        concept = json.load(open("/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data/ConceptNet_VAD_dict.json", "r", encoding="utf-8"))
        
        data_sent['context'] = [sentence]
        data_sent['concepts'] = []
        data_sent['vads'] = []  # each sentence's vad vectors
        data_sent['vad'] = []  # each word's emotion intensity
        
        data_sent['target'] = [[]]
        data_sent['target_vad'] = [[]] # each target word's emotion intensity
        data_sent['target_vads'] = [[]] # each target word's vad vectors
        data_sent['sample_concepts'] = [[]]
        data_sent['emotion'] = [[]]
        data_sent['situation'] = [[]]
        
        test_contexts = data_sent['context']
        # for i, sample in enumerate(test_contexts):
        vads = []  # each item is sentence, each sentence contains a list word' vad vectors
        vad = []
        concepts = []
        total_concepts = []
        total_concepts_tid = []
        for j, sentence in enumerate(test_contexts):
                words_pos = nltk.pos_tag(sentence)

                vads.append(
                    [VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
                vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

                sentence_concepts = [
                    concept[
                        word] if word in word2index and word not in stop_words and word in concept and wordCate(
                        words_pos[wi]) else []
                    for wi, word in enumerate(sentence)]

                sentence_concept_words = []  # for each sentence
                sentence_concept_vads = []
                sentence_concept_vad = []

                for cti, uc in enumerate(sentence_concepts):
                    concept_words = []  # for each token
                    concept_vads = []
                    concept_vad = []
                    if uc != []:
                        for c in uc:
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:
                                if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                    concept_words.append(c[0])
                                    concept_vads.append(VAD[c[0]])
                                    concept_vad.append(emotion_intensity(VAD, c[0]))

                                    total_concepts.append(c[0])
                                    total_concepts_tid.append([j,cti])

                        concept_words = concept_words[:concept_num]
                        concept_vads = concept_vads[:concept_num]
                        concept_vad = concept_vad[:concept_num]

                    sentence_concept_words.append(concept_words)
                    sentence_concept_vads.append(concept_vads)
                    sentence_concept_vad.append(concept_vad)

                sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
                concepts.append(sentence_concepts)

        data_sent['concepts'].append(concepts)
        data_sent['sample_concepts'].append([total_concepts, total_concepts_tid])
        data_sent['vads'].append(vads)
        data_sent['vad'].append(vad)

        test_targets = data_sent['target']
        for i, target in enumerate(test_targets):
            data_sent['target_vads'].append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
            data_sent['target_vad'].append([emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
        print('preprocess finish.')
        
        return data_sent
    
REMOVE_RELATIONS = ["Antonym", "ExternalURL", "NotDesires", "NotHasProperty", "NotCapableOf", "dbpedia", "DistinctFrom", "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo", "SymbolOf", "FormOf", "AtLocation", "DerivedFrom", "SymbolOf", "CreatedBy", "Synonym", "MadeOf"]
    
def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def wordCate(word_pos):
    w_p = get_wordnet_pos(word_pos[1])
    if w_p == wordnet.NOUN or w_p == wordnet.ADV or w_p == wordnet.ADJ or w_p == wordnet.VERB:
        return True
    else:
        return False 

def convert_to_tensor(args,batch_data):
    def merge(sequences):  # len(sequences) = bsz
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1 1=True, in mask means padding.
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    def merge_concept(samples, samples_ext, samples_vads, samples_vad):
        concept_lengths = []  # 每个sample的concepts数目
        token_concept_lengths = []  # 每个sample的每个token的concepts数目
        concepts_list = []
        concepts_ext_list = []
        concepts_vads_list = []
        concepts_vad_list = []

        for i, sample in enumerate(samples):
            length = 0  # 记录当前样本总共有多少个concept，
            sample_concepts = []
            sample_concepts_ext = []
            token_length = []
            vads = []
            vad = []

            for c, token in enumerate(sample):
                if token == []:  # 这个token没有concept
                    token_length.append(0)
                    continue
                length += len(token)
                token_length.append(len(token))
                sample_concepts += token
                sample_concepts_ext += samples_ext[i][c]
                vads += samples_vads[i][c]
                vad += samples_vad[i][c]

            if length > args.total_concept_num:
                value, rank = torch.topk(torch.LongTensor(vad), k=args.total_concept_num)

                new_length = 1
                new_sample_concepts = [args.SEP_idx]  # for each sample
                new_sample_concepts_ext = [args.SEP_idx]
                new_token_length = []
                new_vads = [[0.5,0.0,0.5]]
                new_vad = [0.0]

                cur_idx = 0
                for ti, token in enumerate(sample):
                    if token == []:
                        new_token_length.append(0)
                        continue
                    top_length = 0
                    for ci, con in enumerate(token):
                        point_idx = cur_idx + ci
                        if point_idx in rank:
                            top_length += 1
                            new_length += 1
                            new_sample_concepts.append(con)
                            new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                            new_vads.append(samples_vads[i][ti][ci])
                            new_vad.append(samples_vad[i][ti][ci])
                            assert len(samples_vads[i][ti][ci]) == 3

                    new_token_length.append(top_length)
                    cur_idx += len(token)

                new_length += 1  # for sep token
                new_sample_concepts = [args.SEP_idx] + new_sample_concepts
                new_sample_concepts_ext = [args.SEP_idx] + new_sample_concepts_ext
                new_vads = [[0.5,0.0,0.5]] + new_vads
                new_vad = [0.0] + new_vad

                concept_lengths.append(new_length)  # the number of concepts including SEP
                token_concept_lengths.append(new_token_length)  # the number of tokens which have concepts
                concepts_list.append(new_sample_concepts)
                concepts_ext_list.append(new_sample_concepts_ext)
                concepts_vads_list.append(new_vads)
                concepts_vad_list.append(new_vad)
                assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                assert len(new_token_length) == len(token_length)
            else:
                length += 1
                sample_concepts = [args.SEP_idx] + sample_concepts
                sample_concepts_ext = [args.SEP_idx] + sample_concepts_ext
                vads = [[0.5,0.0,0.5]] + vads
                vad = [0.0] + vad

                concept_lengths.append(length)
                token_concept_lengths.append(token_length)
                concepts_list.append(sample_concepts)
                concepts_ext_list.append(sample_concepts_ext)
                concepts_vads_list.append(vads)
                concepts_vad_list.append(vad)

        if max(concept_lengths) != 0:
            padded_concepts = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len); add 1 for root
            padded_concepts_ext = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths), 1) ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(len(samples), max(concept_lengths))  ## padding index 1 (bsz, max_concept_len)
            padded_mask = torch.ones(len(samples), max(concept_lengths)).long()  # concept(dialogue) state

            for j, concepts in enumerate(concepts_list):
                end = concept_lengths[j]
                if end == 0:
                    continue
                padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                padded_mask[j, :end] = args.KG_idx  # for DIALOGUE STATE

            return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
        else:  # there is no concept in this mini-batch
            return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])

    def merge_vad(vads_sequences, vad_sequences):  # for context
        lengths = [len(seq) for seq in vad_sequences]
        padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
        padding_vad = torch.FloatTensor([[0.5]]).repeat(len(vads_sequences), max(lengths))

        for i, vads in enumerate(vads_sequences):
            end = lengths[i]  # the length of context
            padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
            padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
        return padding_vads, padding_vad  # (bsz, max_context_len, 3); (bsz, max_context_len)

    def adj_mask(context, context_lengths, concepts, token_concept_lengths):
        '''

        :param self:
        :param context: (bsz, max_context_len)
        :param context_lengths: [] len=bsz
        :param concepts: (bsz, max_concept_len)
        :param token_concept_lengths: [] len=bsz;
        :return:
        '''
        bsz, max_context_len = context.size()
        max_concept_len = concepts.size(1)  # include sep token
        adjacency_size = max_context_len + max_concept_len
        adjacency = torch.ones(bsz, max_context_len, adjacency_size)   ## todo padding index 1, 1=True

        for i in range(bsz):
            # ROOT -> TOKEN
            adjacency[i, 0, :context_lengths[i]] = 0
            adjacency[i, :context_lengths[i], 0] = 0

            con_idx = max_context_len+1       # add 1 because of sep token
            for j in range(context_lengths[i]):
                adjacency[i, j, j - 1] = 0 # TOEKN_j -> TOKEN_j-1

                token_concepts_length = token_concept_lengths[i][j]
                if token_concepts_length == 0:
                    continue
                else:
                    adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                    adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                    con_idx += token_concepts_length
        return adjacency

    # batch_data.sort(key=lambda x: len(x["context"]), reverse=True)
    batch_data = [batch_data]
    item_info = {}
    for key in batch_data[0].keys():
        item_info[key] = [d[key] for d in batch_data]

    assert len(item_info['context']) == len(item_info['vad'])

    ## dialogue context
    context_batch, context_lengths = merge(item_info['context'])
    context_ext_batch, _ = merge(item_info['context_ext'])
    mask_context, _ = merge(item_info['context_mask'])  # for dialogue state!

    ## dialogue context vad
    context_vads_batch, context_vad_batch = merge_vad(item_info['vads'], item_info['vad'])  # (bsz, max_context_len, 3); (bsz, max_context_len)

    assert context_batch.size(1) == context_vad_batch.size(1)


    ## concepts, vads, vad
    concept_inputs = merge_concept(item_info['concept'],
                                    item_info['concept_ext'],
                                    item_info["concept_vads"],
                                    item_info["concept_vad"])  # (bsz, max_concept_len)
    concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = concept_inputs

    ## adja_mask (bsz, max_context_len, max_context_len+max_concept_len)
    if concept_batch.size()[0] != 0:
        adjacency_mask_batch = adj_mask(context_batch, context_lengths, concept_batch, token_concept_lengths)
    else:
        adjacency_mask_batch = torch.Tensor([])

    ## target response
    target_batch, target_lengths = merge(item_info['target'])
    target_ext_batch, _ = merge(item_info['target_ext'])

    d = {}
    ##input
    d["context_batch"] = context_batch.to(args.device)  # (bsz, max_context_len)
    d["context_ext_batch"] = context_ext_batch.to(args.device)  # (bsz, max_context_len)
    d["context_lengths"] = torch.LongTensor(context_lengths).to(args.device)  # (bsz, )
    d["mask_context"] = mask_context.to(args.device)
    d["context_vads"] = context_vads_batch.to(args.device)   # (bsz, max_context_len, 3)
    d["context_vad"] = context_vad_batch.to(args.device)  # (bsz, max_context_len)

    ##concept
    d["concept_batch"] = concept_batch.to(args.device)  # (bsz, max_concept_len)
    d["concept_ext_batch"] = concept_ext_batch.to(args.device)  # (bsz, max_concept_len)
    d["concept_lengths"] = torch.LongTensor(concept_lengths).to(args.device)  # (bsz)
    d["mask_concept"] = mask_concept.to(args.device)  # (bsz, max_concept_len)
    d["concept_vads_batch"] = concepts_vads_batch.to(args.device)  # (bsz, max_concept_len, 3)
    d["concept_vad_batch"] = concepts_vad_batch.to(args.device)   # (bsz, max_concept_len)
    d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(args.device)

    ##output
    d["target_batch"] = target_batch.to(args.device)  # (bsz, max_target_len)
    d["target_ext_batch"] = target_ext_batch.to(args.device)
    d["target_lengths"] = torch.LongTensor(target_lengths).to(args.device)  # (bsz,)

    ##program
    # d["target_emotion"] = torch.LongTensor(item_info['emotion']).to(args.device)
    # d["emotion_label"] = torch.LongTensor(item_info['emotion_label']).to(args.device)  # (bsz,)
    # d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(args.device)
    # assert d["emotion_widx"].size() == d["emotion_label"].size()

    ##text
    d["context_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["emotion_txt"] = item_info['emotion_text']
    d["concept_txt"] = item_info['concept_text']
    d["oovs"] = item_info["oovs"]

    return d
if __name__ == "__main__":
    print("end")
