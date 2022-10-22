# reference: https://github.com/chujiezheng/DiffKS/blob/master/prepare_holl_data.py

import json
import os.path
from collections import defaultdict
import nltk
import random
from tqdm import tqdm
import numpy as np
from nltk.tokenize import WordPunctTokenizer

func = lambda sen: ' '.join(WordPunctTokenizer().tokenize(sen.strip())).lower()


def _norm1(sent):
    return ' '.join(sent.strip().split())


def _norm2(sent):
    return ' '.join(nltk.word_tokenize(_norm1(sent))).lower()


def nltk2sentence(tokens):
    sent = " ".join(tokens)
    out_string = sent.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(" ' ", "'"). \
        replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string


def _norm3(sent):
    return nltk2sentence(sent.strip().split())


def split_full(full: str, span: str, span_st):
    if span == '' or len(nltk.word_tokenize(span.strip())) < 2:
        return None

    if span_st is None:
        span_st = full.find(span)
        if span_st == -1:
            return None
    try:
        assert full[span_st:][:len(span)] == span
    except:
        return None

    before = _norm3(full[:span_st].replace('EOD', '').strip())
    after = _norm3(full[span_st:][len(span):].replace('EOD', '').strip())
    span = _norm3(span.replace('EOD', '').strip())

    before_sentences = nltk.sent_tokenize(before)
    after_sentences = nltk.sent_tokenize(after)
    all_sentences = nltk.sent_tokenize(before + ' ' + span + ' ' + after)

    bdx, adx = None, None
    before = []
    after = []
    if len(before_sentences) > 0:
        bdx = 0
        while before_sentences[bdx] == all_sentences[bdx]:
            bdx += 1
            if bdx == len(before_sentences):
                break
        before = all_sentences[:bdx]
    if len(after_sentences) > 0:
        adx = 0
        while after_sentences[-1 - adx] == all_sentences[-1 - adx]:
            adx += 1
            if adx == len(after_sentences):
                break
        after = all_sentences[-adx:]
        span = all_sentences[bdx: -adx]
    else:
        span = all_sentences[bdx:]

    before = sorted(set(before) - set(span), key=lambda x: before.index(x))
    after = sorted(set(after) - set(span), key=lambda x: after.index(x))

    for sents in [before, span, after]:
        for i in range(len(sents)):
            sents[i] = func(sents[i])
    span = ' '.join(span)

    return before + [span] + after, len(before)

if __name__ == '__main__':
    root_path = '/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/holl_e/raw_data/'
    original_data_path = os.path.join(root_path,'original_data')
    processed_data_path = root_path

    if not os.path.exists(original_data_path):
        raise FileNotFoundError("Original Holl-E dataset are not found in {}!"
                                "Please download the Holl-E dataset following the instructions in readme or specify "
                                "your own dataset path!".format(original_data_path))

    keys = ['dev', 'test',
            'train', ]
    for type_name in ['full']:  # , 'oracle_reduced', 'mixed', 'full', 'full_reduced']:
        print(type_name)
        for key in keys:
            know_num = []

            main2raw = defaultdict(list)
            raw_data = {}
            with open(os.path.join(original_data_path,f'raw_data/{key}_data.json')) as f:
                for each in json.load(f):
                    data = each.copy()
                    del data['all_documents']
                    del data['full_history']
                    del data['short_history']

                    data['know'] = data[type_name]
                    data['span_st'] = data['answer_start_' + type_name]

                    del data['oracle_reduced']
                    del data['oracle']
                    del data['mixed']
                    del data['full']
                    del data['full_reduced']

                    del data['response_start']
                    del data['answer_start_oracle']
                    del data['answer_start_oracle_reduced']
                    del data['answer_start_full']
                    del data['answer_start_full_reduced']
                    del data['answer_start_mixed']

                    raw_data[data['example_id']] = data
                    main2raw[data['chat_id']].append(data['example_id'])

            if key == 'test':
                with open(os.path.join(original_data_path, 'experiment_data/multi_reference_test.json')) as f:
                    mf_data = json.load(f)
                    for example_id in mf_data.keys():
                        all_responses = list(set(mf_data[example_id]['responses']))
                        if len(all_responses) == 1:
                            continue
                        ori_response = raw_data[example_id]['response']
                        raw_data[example_id]['response'] = response = [ori_response]
                        for each in all_responses:
                            if each != ori_response:
                                response.append(each)

            span_diff_cnt = 0
            main_data = []
            with open(os.path.join(original_data_path, f'main_data/{key}_data.json')) as f:
                f = json.load(f)
                for each in tqdm(f, total=len(f)):
                    main_data.append({})
                    data = main_data[-1]
                    post = data['post'] = []
                    resp = data['resp'] = []
                    know = data['know'] = []
                    atten = data['atten'] = []

                    example_list = main2raw[each['chat_id']]
                    sorted(example_list)
                    chat = each['chat']
                    # only consider the first 10 + 10 turns
                    main_post = list(chat[i * 2] for i in range(len(chat) // 2))[:10]
                    main_resp = list(chat[i * 2 + 1] for i in range(len(chat) // 2))[:10]
                    main_spans = list(each['spans'][i * 2 + 1] for i in range(len(chat) // 2))[:10]

                    raw_post = list(raw_data[e]['query'] for e in example_list)
                    raw_resp = list(raw_data[e]['response'] for e in example_list)
                    raw_know = list(raw_data[e]['know'] for e in example_list)
                    raw_spans = list(raw_data[e]['span'] for e in example_list)
                    raw_span_st = list(raw_data[e]['span_st'] for e in example_list)

                    main_turn = 0
                    raw_turn = 0
                    while main_turn < len(main_resp):
                        if raw_turn == len(raw_resp) or \
                                main_resp[main_turn] != (
                        raw_resp[raw_turn][0] if isinstance(raw_resp[raw_turn], list) else raw_resp[raw_turn]):
                            post.append(_norm2(main_post[main_turn]))
                            resp.append(_norm2(main_resp[main_turn]))
                            atten.append(-1)
                            know.append(None)
                            main_turn += 1
                            continue

                        assert main_post[main_turn] == raw_post[raw_turn]
                        post.append(_norm2(main_post[main_turn]))
                        if isinstance(raw_resp[raw_turn], list):
                            resp.append(list(map(_norm2, raw_resp[raw_turn])))
                        else:
                            resp.append(_norm2(main_resp[main_turn]))

                        if main_spans[main_turn] != raw_spans[raw_turn]:
                            span_diff_cnt += 1
                        processed_result = split_full(raw_know[raw_turn], raw_spans[raw_turn], raw_span_st[raw_turn])
                        if processed_result is not None:
                            processed_know, processed_atten = processed_result
                            know.append(processed_know)
                            atten.append(processed_atten)
                        else:
                            know.append(None)
                            atten.append(-1)
                        main_turn += 1
                        raw_turn += 1

                    assert len(post) == len(resp) == len(know) == len(atten)
                    # handle cases where know is None
                    if know[0] is None:
                        i = 1
                        while i < len(know) and know[i] is None:
                            i += 1
                        if i == len(know):
                            main_data.pop()
                            # print(each['chat_id'], 'knowledge corrupted')
                            continue
                            # raise ValueError('knowledge corrupted')
                        while i > 0:
                            know[i - 1] = know[i].copy()
                            i -= 1

                    last_not_none = know[0]
                    know_num.append(len(know[0]))
                    for i in range(1, len(know)):
                        if know[i] is not None:
                            last_not_none = know[i]
                        else:
                            know[i] = last_not_none
                        know_num.append(len(know[i]))

                    assert all([e is not None for e in know])

            # print('\t', key, span_diff_cnt)
            print('\t', np.mean(know_num), np.std(know_num))
            with open(os.path.join(processed_data_path, f'{key}_data.json'),'w') as f:
                json.dump(main_data, f, indent=4, ensure_ascii=False)