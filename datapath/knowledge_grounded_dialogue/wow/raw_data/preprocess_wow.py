import sys

import json
import os
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk import word_tokenize

if not os.path.exists("./original/prepared_data"):
    os.mkdir("./original/prepared_data")

# save_names = ['train', 'test-seen', 'test-unseen', 'dev-seen', 'dev-unseen']
file_names = ["test", "test_unseen", "valid", "valid_unseen", "train", ]
save_names = ["test_random_split", "test_topic_split", "valid_random_split", "valid_topic_split", "train", ]

f = lambda sen: ' '.join(WordPunctTokenizer().tokenize(sen.strip())).lower()
# f = lambda sen: ' '.join(word_tokenize(sen)).lower()

for key, name in zip(file_names, save_names):
    print(key)
    total_data = []

    # d = json.load(open("./Wizard-of-Wikipedia/%s_collected.json" % key, 'r'))
    d = json.load(open("./original/%s_collected.json" % key, 'r'))
    for data in tqdm(d, total=len(d)):
        new_data = {}
        new_data['chosen_topics'] = data['topics']
        new_data['posts'] = list(map(f, data['post']))
        # new_data['posts'][0] = ' '.join(new_data['posts'][0].split()[1:])
        new_data['responses'] = list(map(f, data['response']))
        assert all(e[0] == 'no_passages_used __knowledge__ no_passages_used' for e in data['knowledge'])
        new_data['knowledge'] = [['no_knowledge_used'] + list(map(lambda x: f(x.split('__knowledge__')[1]), e[1:])) for
                                 e in data['knowledge']]
        new_data['labels'] = data['labels']
        total_data.append(new_data)

    json.dump(total_data, open('./original/prepared_data/%s.json' % name, 'w'), indent=4, ensure_ascii=False,
              sort_keys=True)

with open('./original/prepared_data/dev.json', 'w') as f:
    data = json.load(open('./original/prepared_data/valid_random_split.json')) + \
           json.load(open('./original/prepared_data/valid_topic_split.json'))
    json.dump(data, f, ensure_ascii=False, indent=4)
    # os.remove('valid_random_split.json')
    # os.remove('valid_topic_split.json')

with open('./original/prepared_data/test_seen.json', 'w') as f:
    data = json.load(open('./original/prepared_data/test_random_split.json'))
    json.dump(data, f, ensure_ascii=False, indent=4)
    # os.remove('test_random_split.json')

with open('./original/prepared_data/test_unseen.json', 'w') as f:
    data = json.load(open('./original/prepared_data/test_topic_split.json'))
    json.dump(data, f, ensure_ascii=False, indent=4)
    # os.remove('test_topic_split.json')
os.system("mv ./original/prepared_data/* ./")
print("All the files are saved to {}!".format(os.path.abspath("./")))