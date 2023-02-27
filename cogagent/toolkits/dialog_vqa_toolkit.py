from cogagent.toolkits.base_toolkit import BaseToolkit
import json
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY = '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'
import csv
import numpy as np
from tqdm import tqdm
import random
import openai  # pip install --upgrade openai
import time
import clip  # pip install git+https://github.com/openai/CLIP.git
import requests
from PIL import Image
from transformers import pipeline
from skimage import io  # pip3 install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple/


class DialogVQAToolkit(BaseToolkit):
    """
    & 功能
    输入一张图片和一个问题，返回问题的简短答案和句子答案
    infer_valid封闭域推断现有数据集
    prepare_feature开放域准备原始数据
    infer_one开放域推断新的一条数据
    & 参数
    dataset_name:训练的数据集名称
    data_path:数据集路径
    model_name:模型的名字
    language:语言
    n_shot：n_shot
    n_ensemble:集成n_ensemble组n_shot
    & 数据集说明
    coco_annotations
      --captions_train2014.json
      --captions_val2014.json
      --mscoco_train2014_annotations.json
      --mscoco_val2014_annotations.json()
      --OpenEnded_mscoco_train2014_questions.json
      --OpenEnded_mscoco_val2014_questions.json()

    coco_clip_new
      --okvqa_qa_line2sample_idx_train2014.json
      --okvqa_qa_line2sample_idx_val2014.json

    input_text
      --coco_caption_pred_tags
      ----test.score.json.lineidx
      ----test.score.json.tsv
      ----train.score.json.lineidx
      ----train.score.json.tsv
      ----val.score.json.lineidx
      ----val.score.json.tsv
      --vinvl_caption
      ----VinVL_base_test.tsv
      ----VinVL_base_val.tsv
      ----VinVL_base_val2014.tsv

    例子：
    "image_id": 297147,
    "question_id": 2971475
    "question": "What sport can you use this for?",
    "image_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000297147.jpg"

    & 待做
    把答案组织成一个句子


    """

    def __init__(self,
                 dataset_name,
                 model_name,
                 data_path,
                 language,
                 n_shot=16,
                 n_ensemble=1):
        super().__init__()
        self.dataset_name = dataset_name,
        self.model_name = model_name,
        self.data_path = data_path,
        self.language = language
        self.n_shot = n_shot
        self.n_ensemble = n_ensemble
        self.n = self.n_shot * self.n_ensemble

        self.apikey = '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'
        self.engine = 'davinci'

        self.train_answer_anno_file = os.path.join(data_path, "coco_annotations/mscoco_train2014_annotations.json")
        self.train_question_anno_file = os.path.join(data_path,
                                                     'coco_annotations/OpenEnded_mscoco_train2014_questions.json')
        self.train_caption_anno_file = os.path.join(data_path, 'coco_annotations/captions_train2014.json')
        self.val_answer_anno_file = os.path.join(data_path, "coco_annotations/mscoco_val2014_annotations.json")
        self.val_question_anno_file = os.path.join(data_path,
                                                   'coco_annotations/OpenEnded_mscoco_val2014_questions.json')
        self.val_caption_anno_file = os.path.join(data_path, 'input_text/vinvl_caption/VinVL_base_val2014.tsv')
        self.train_file = os.path.join(data_path, 'coco_clip_new/okvqa_qa_line2sample_idx_train2014.json')
        self.val_file = os.path.join(data_path, 'coco_clip_new/okvqa_qa_line2sample_idx_val2014.json')
        self.train_question_feature_file = os.path.join(data_path,
                                                        'coco_clip_new/coco_clip_vitb16_train2014_okvqa_question.npy')
        self.train_image_feature_file = os.path.join(data_path,
                                                     'coco_clip_new/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy')
        self.valid_question_feature_file = os.path.join(data_path,
                                                        'coco_clip_new/coco_clip_vitb16_val2014_okvqa_question.npy')
        self.valid_image_feature_file = os.path.join(data_path,
                                                     'coco_clip_new/coco_clip_vitb16_val2014_okvqa_convertedidx_image.npy')

        self.train_question_open_domian_feature_file = os.path.join(data_path,
                                                                    'coco_clip_newnew/coco_clip_vitb16_train2014_okvqa_question.npy')
        self.train_image_open_domian_feature_file = os.path.join(data_path,
                                                                 'coco_clip_newnew/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy')

        self.train_answer_anno = json.load(open(self.train_answer_anno_file, 'r'))
        self.train_question_anno = json.load(open(self.train_question_anno_file, 'r'))
        self.val_answer_anno = json.load(open(self.val_answer_anno_file, 'r'))
        self.val_question_anno = json.load(open(self.val_question_anno_file, 'r'))

        self.train_answer_dict = {}
        for sample in self.train_answer_anno['annotations']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in self.train_answer_dict:
                self.train_answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [x['answer'] for
                                                                                                        x in
                                                                                                        sample[
                                                                                                            'answers']]
        self.val_answer_dict = {}
        for sample in self.val_answer_anno['annotations']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in self.val_answer_dict:
                self.val_answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [x['answer'] for x
                                                                                                      in
                                                                                                      sample['answers']]
        # {image_id<->question_id:候选答案列表}
        # {'297147<->2971475':['race', 'race', 'race', 'race', 'race', 'race', 'motocross', 'motocross', 'ride', 'ride']}

        self.train_question_dict = {}
        for sample in self.train_question_anno['questions']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in self.train_question_dict:
                self.train_question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample[
                    'question']
        self.val_question_dict = {}
        for sample in self.val_question_anno['questions']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in self.val_question_dict:
                self.val_question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample[
                    'question']
        # {image_id<->question_id: 问题字符串}
        # {'297147<->2971475':'What sport can you use this for?'}

        self.val_keys = list(self.val_question_dict.keys())
        # [image_id<->question_id]的列表，是验证集的列表

        coco_caption = json.load(open(self.train_caption_anno_file, 'r'))['annotations']
        self.train_caption_dict = {}
        for sample in coco_caption:
            if sample['image_id'] not in self.train_caption_dict:
                self.train_caption_dict[sample['image_id']] = [sample['caption']]
            else:
                self.train_caption_dict[sample['image_id']].append(sample['caption'])
        # {image_id:[caption列表]}字典

        self.val_caption_dict = {}
        val_read_tsv = csv.reader(open(self.val_caption_anno_file, 'r'), delimiter="\t")
        for row in val_read_tsv:
            if int(row[0]) not in self.val_caption_dict:
                self.val_caption_dict[int(row[0])] = [row[1].split('caption": "')[1].split('", "conf"')[0]]
            else:
                self.val_caption_dict[int(row[0])].append(row[1].split('caption": "')[1].split('", "conf"')[0])
        # {image_id:[caption列表]}字典

        self.train_idx2key = json.load(open(self.train_file, 'r'))
        self.val_idx2key = json.load(open(self.val_file, 'r'))
        # {训练验证编号:image_id<->question_id}(编号是特征矩阵的顺序)
        self.val_key2idx = {}
        for ii in self.val_idx2key:
            self.val_key2idx[self.val_idx2key[ii]] = int(ii)
        # {image_id<->question_id:验证编号}

        self.train_question_feature = np.load(self.train_question_feature_file)
        self.train_image_feature = np.load(self.train_image_feature_file)
        self.valid_question_feature = np.load(self.valid_question_feature_file)
        self.valid_image_feature = np.load(self.valid_image_feature_file)
        self.train_question_open_domian_feature = np.load(self.train_question_open_domian_feature_file)
        self.train_image_open_domian_feature = np.load(self.train_image_open_domian_feature_file)
        # 问题特征和图像特征

    def prepare_feature(self):
        device = "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image_file = json.load(open(self.train_caption_anno_file, 'r'))["images"]
        image_id2url = {}
        for content in image_file:
            image_id2url[content["id"]] = content['coco_url']
        if not os.path.exists(os.path.join(self.data_path[0], 'coco_clip_newnew')):
            os.makedirs(os.path.join(self.data_path[0], 'coco_clip_newnew'))

        text_features_list = []
        for sample in tqdm(self.train_question_anno['questions']):
            question = sample["question"]
            text = clip.tokenize([question]).to(device)
            text_features = model.encode_text(text).squeeze().detach().numpy()
            text_features_list.append(text_features)
        text_features_matrix = np.array(text_features_list)
        path = os.path.join(self.data_path[0],
                            'coco_clip_newnew/coco_clip_vitb16_train2014_okvqa_question.npy')
        np.save(path, text_features_matrix, allow_pickle=True, fix_imports=True)

        image_features_list = []
        for index, id_pair in tqdm(self.train_idx2key.items()):
            image_id, text_id = id_pair.split('<->')
            image_id = int(image_id)
            url = image_id2url[image_id]
            image = Image.open(requests.get(url, stream=True).raw)
            image = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image).squeeze().detach().numpy()
            image_features_list.append(image_features)
        image_features_matrix = np.array(image_features_list)
        path = os.path.join(self.data_path[0],
                            'coco_clip_newnew/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy')
        np.save(path, image_features_matrix, allow_pickle=True, fix_imports=True)

    def infer_one(self, raw_dict,image_str=True):
        infer_dict = {}

        device = "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        question = raw_dict["question"]
        text = clip.tokenize([question]).to(device)
        text_features = model.encode_text(text).squeeze().detach().numpy()

        url = raw_dict["image_url"]
        if image_str:
            if url[:4]=="http":
                image = Image.open(requests.get(url, stream=True).raw)
            else:
                image = Image.open(url)
        else:
            image=url
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image).squeeze().detach().numpy()

        question_similarity = np.matmul(self.train_question_open_domian_feature, text_features)
        image_similarity = np.matmul(self.train_image_open_domian_feature, image_features)

        image_to_text = pipeline("image-to-text",
                                 model="nlpconnect/vit-gpt2-image-captioning",
                                 max_new_tokens=50)
        caption_i = image_to_text(url)[0]['generated_text']

        similarity = question_similarity + image_similarity
        index = similarity.argsort()[-self.n:][::-1]  # 相似度从小到大排序并且逆序
        context_key_list = [self.train_idx2key[str(x)] for x in index]

        pred_answer_list, pred_prob_list = [], []
        for repeat in range(self.n_ensemble):
            prompt = 'Please answer the question according to the above context.\n===\n'
            for ni in range(self.n_shot):
                context_key = context_key_list[ni + self.n_shot * repeat]
                img_context_key = int(context_key.split('<->')[0])
                prompt += 'Context: %s\n===\n' % self.train_caption_dict[img_context_key][
                    random.randint(0, len(self.train_caption_dict[img_context_key]) - 1)]
                prompt += 'Q: %s\nA: %s\n\n===\n' % (
                    self.train_question_dict[context_key], self.train_answer_dict[context_key][0])
            prompt += 'Context: %s\n===\n' % caption_i
            prompt += 'Q: %s\nA:' % question
            response = None

            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=30,
                    logprobs=1,
                    temperature=0.,
                    stream=False,
                    stop=["\n", "<|endoftext|>"]
                )
            except Exception as e:
                print(e)
                exit(0)
            plist = []
            for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                if response['choices'][0]['logprobs']['tokens'][ii] == '\n':
                    break
                plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
            pred_answer_list.append(self.process_answer(response['choices'][0]["text"]))  # 预测答案列表
            pred_prob_list.append(sum(plist))  # 预测概率列表

        maxval = -999.
        # 滤除概率太低的
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]

        prompt_sentence = caption_i + '.' + question + pred_answer + "." + "This is because"
        response_sentence = openai.Completion.create(
            engine=self.engine,
            prompt=prompt_sentence,
            max_tokens=50,
            temperature=0,
            frequency_penalty=1.2,
            presence_penalty=1.2,
            stop=["\n"]
        )
        reason = response_sentence['choices'][0]["text"]

        infer_dict["reason"] = reason
        infer_dict["pred_answer"] = pred_answer
        infer_dict["caption"] = caption_i
        return infer_dict

    def infer_valid(self, infer_num=999999999):
        answers = []
        for index, key in tqdm(enumerate(self.val_keys), total=infer_num):
            if index < infer_num:
                answers.append(self.sample_inference(key))

        infer_dict = {}
        infer_valid_list = []
        acc = 0.
        for answer in answers:
            infer_item_dict = {}
            infer_item_dict["question_image__id"] = answer[0]
            infer_item_dict["answer"] = answer[1]
            infer_item_dict["prompt"] = answer[2]
            infer_item_dict["acc"] = answer[3]
            acc += float(answer[3])
            infer_valid_list.append(infer_item_dict)
        acc = acc * 100. / len(answers)
        infer_dict["acc"] = acc
        infer_dict["infer_valid_list"] = infer_valid_list
        return infer_dict

    def sample_inference(self, key):
        img_key = int(key.split('<->')[0])
        question, answer, caption = self.val_question_dict[key], self.val_answer_dict[key], self.val_caption_dict[
            img_key]
        caption_i = caption[random.randint(0, len(caption) - 1)]  # 如果存在多个caption则随机选择其中一个
        lineid = self.val_key2idx[key]
        question_similarity = np.matmul(self.train_question_feature, self.valid_question_feature[lineid, :])
        image_similarity = np.matmul(self.train_image_feature, self.valid_image_feature[lineid, :])
        similarity = question_similarity + image_similarity
        index = similarity.argsort()[-self.n:][::-1]  # 相似度从小到大排序并且逆序
        context_key_list = [self.train_idx2key[str(x)] for x in index]

        pred_answer_list, pred_prob_list = [], []
        for repeat in range(self.n_ensemble):
            time.sleep(2)
            prompt = 'Please answer the question according to the above context.\n===\n'
            for ni in range(self.n_shot):
                context_key = context_key_list[ni + self.n_shot * repeat]
                img_context_key = int(context_key.split('<->')[0])
                prompt += 'Context: %s\n===\n' % self.train_caption_dict[img_context_key][
                    random.randint(0, len(self.train_caption_dict[img_context_key]) - 1)]
                prompt += 'Q: %s\nA: %s\n\n===\n' % (
                    self.train_question_dict[context_key], self.train_answer_dict[context_key][0])
            prompt += 'Context: %s\n===\n' % caption_i
            prompt += 'Q: %s\nA:' % question
            response = None

            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=5,
                    logprobs=1,
                    temperature=0.,
                    stream=False,
                    stop=["\n", "<|endoftext|>"]
                )
            except Exception as e:
                print(e)
                exit(0)
            plist = []
            for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                if response['choices'][0]['logprobs']['tokens'][ii] == '\n':
                    break
                plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
            pred_answer_list.append(self.process_answer(response['choices'][0]["text"]))  # 预测答案列表
            pred_prob_list.append(sum(plist))  # 预测概率列表

        maxval = -999.
        # 滤除概率太低的
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]
        # 数和候选答案一样的有几个
        counter = 0
        for ii in range(len(answer)):
            if pred_answer == answer[ii]: counter += 1
        return (key, pred_answer, prompt, min(1., float(counter) * 0.3))

    def process_answer(self, answer):
        answer = answer.replace('.', '').replace(',', '').lower()
        to_be_removed = {'a', 'an', 'the', 'to', ''}
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return ' '.join(answer_list)


if __name__ == "__main__":
    raw_dict = {"image_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000517985.jpg",
                "question": "Which essential food group is missing?"}

    # http://images.cocodataset.org/val2014/COCO_val2014_000000297147.jpg
    # What sport can you use this for?
    # What is its color?
    # What is its speed?
    # What job often ride it?

    # http://images.cocodataset.org/train2014/COCO_train2014_000000392136.jpg

    # http://images.cocodataset.org/train2014/COCO_train2014_000000384029.jpg
    # Who likes to eat the things in the picture?
    image = io.imread("http://images.cocodataset.org/train2014/COCO_train2014_000000517985.jpg")
    io.imshow(image)
    io.show()
    dialogvqatoolkit = DialogVQAToolkit(
        dataset_name="OKVQA",
        model_name="pica",
        data_path="/data/mentianyi/code/CogNLP/datapath/vqa/okvqa",
        language="english"
    )
    # 推断现有数据集
    # infer_valid_dict = dialogvqatoolkit.infer_valid(infer_num=100)
    # 准备数据
    dialogvqatoolkit.prepare_feature()
    # 推断单条数据
    infer_dict = dialogvqatoolkit.infer_one(raw_dict=raw_dict,image_str=True)
    print(infer_dict)
