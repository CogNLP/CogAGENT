from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
import numpy as np
transformers.logging.set_verbosity_error()  # set transformers logging level


class HollEForDiffksProcessor(BaseProcessor):
    def __init__(self, max_token_len, vocab,debug=False):
        super().__init__(debug)
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = None

        self.word_vocab = vocab["word_vocab"]
        self.word2id= self.word_vocab.get_label2id_dict()
        self.unk_id,self.eos_id = self.word2id["<unk>"],self.word2id["<eos>"]
        self.line2id = lambda line: ([] + list(map(lambda word: self.word2id.get(word, self.unk_id), line)) + [self.eos_id])[:self.max_token_len]
        self.know2id = lambda line: (list(map(lambda word: self.word2id.get(word, self.unk_id), line)))[:self.max_token_len]

        self._max_context_length = 100
        self.max_sent_num = 10
        self.max_wiki_num = 200
        self.max_post_length = 100
        self.max_resp_length = 100
        self.max_wiki_length = 100

        self.valid_vocab_len = 16004

    def _process(self, data,is_training=True):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for dialog in tqdm(data):
            post,resp,wiki,atten = dialog

            # construct inputs to the models
            post = list(map(self.line2id, post))
            resp = list(map(self.line2id, resp))
            wiki = [list(map(self.know2id,wiki_psg)) for wiki_psg in wiki]

            if is_training:
                for j in range(len(post)):
                    if j == 0:
                        continue
                    post[j] = (post[j - 1] + resp[j - 1] + post[j])[-self._max_context_length:]

            sent_num = len(post)

            post_length = list(map(len,post))[:self.max_sent_num]
            post_length += [0] * (self.max_sent_num - len(post_length))

            resp_length = list(map(len,resp))[:self.max_sent_num]
            resp_length += [0] * (self.max_sent_num - len(resp_length))

            wiki_num = list(map(len,wiki))[:self.max_sent_num]
            wiki_num += [0] * (self.max_sent_num - len(wiki_num))

            atten = atten[:self.max_sent_num]
            atten += [0] * (self.max_sent_num - len(atten))

            wiki_length = np.zeros((self.max_sent_num,self.max_wiki_num),dtype=int)
            for i in range(sent_num):
                single_wiki_length = list(map(len,wiki[i]))[:self.max_wiki_num]
                wiki_length[i,:len(single_wiki_length)] = single_wiki_length

            wiki_length = wiki_length.tolist()

            padded_post = np.zeros((self.max_sent_num,self.max_post_length),dtype=int)
            padded_resp = np.zeros((self.max_sent_num,self.max_resp_length),dtype=int)
            for i in range(sent_num):
                single_post = post[i][:self.max_post_length]
                single_resp = resp[i][:self.max_resp_length]
                padded_post[i,:len(single_post)] = single_post
                padded_resp[i,:len(single_resp)] = single_resp

            print("Debug Usage")
            padded_wiki = np.zeros((self.max_sent_num,self.max_wiki_num,self.max_wiki_length),dtype=int)
            for i in range(sent_num):
                for j in range(wiki_num[i]):
                    single_wiki = wiki[i][j][:self.max_wiki_length]
                    padded_wiki[i,j,:len(single_wiki)] = single_wiki

            post_all_vocabs = padded_post
            resp_all_vocabs = padded_resp

            padded_post[padded_post >= self.valid_vocab_len] = self.unk_id
            padded_resp[padded_resp >= self.valid_vocab_len] = self.unk_id
            padded_wiki[padded_wiki >= self.valid_vocab_len] = self.unk_id

            post = padded_post
            resp = padded_resp
            wiki = padded_wiki

            datable("sent_num",sent_num)
            datable("post_length",post_length)
            datable("resp_length",resp_length)
            datable("wiki_num",wiki_num)
            datable("wiki_length",wiki_length)
            datable("post",post.tolist())
            datable("resp",resp.tolist())
            datable("atten",atten)
            datable("wiki",wiki.tolist())
            datable("post_all_vocabs",post_all_vocabs.tolist())
            datable("resp_all_vocabs",resp_all_vocabs.tolist())

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data,is_training=False)

    def process_test(self, data):
        return self._process(data,is_training=False)


if __name__ == "__main__":
    cache_file = "/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/holl_e/cache/reader_datas.pkl"
    from cogagent.utils.io_utils import load_pickle
    train_data,dev_data,test_data,vocab = load_pickle(cache_file)

    processor = HollEForDiffksProcessor(max_token_len=512, vocab=vocab,debug=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")

