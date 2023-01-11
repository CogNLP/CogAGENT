import csv
import json
import random
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, Tuple

from tqdm import tqdm
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")
from random import shuffle

def _tokenize(sent: str) -> str:
    return " ".join([tok.text for tok in nlp(sent)])


def read_csv(data_file: str) -> Iterable[Tuple[str, int]]:
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header row
        dialog_id = 0
        for i, row in enumerate(reader):
            dialog_id += 1
            dialogue, _, _ = row[0].strip(), row[1].strip(), row[2].strip()

            yield dialogue, dialog_id


def parse_message(dialogue: str, dialog_id: int) -> Iterable[Dict[str, Any]]:
    json_dialog = json.loads(dialogue)
    history = []
    metadata = {}
    for i, turn in enumerate(json_dialog):
        if i == 0:
            if "message" in turn:
                history.append(_tokenize(turn["message"]))
        else:
            if "metadata" in turn:
                if "path" in turn["metadata"]:
                    metadata = {
                        "paths": turn["metadata"]["path"][1],
                        "render": turn["metadata"]["path"][2],
                    }
            else:
                response = _tokenize(turn["message"])
                yield {
                    "history": history,
                    "response": response,
                    "speaker": turn["sender"],
                    "knowledge_base": metadata,
                    "dialogue_id": dialog_id,
                }

                metadata = {}
                history.append(response)


def convert(data_file: str, out_file: list):
    row_count = sum(1 for row in read_csv(data_file))
    with tqdm(total = row_count) as pbar:
        raw_lines = [item for item in read_csv(data_file)]
        shuffle(raw_lines)
        train_and_dev,test = train_test_split(raw_lines,test_size=0.1)
        train,dev = train_test_split(train_and_dev,test_size=1/9)

        for target_lines,target_file,name in zip([train,dev,test],out_file,['train','dev','test']):
            lines = []
            pbar.set_description('Processing {}'.format(name))
            for dialogue, dialog_id in target_lines:
                for utterance in parse_message(dialogue, dialog_id):
                    lines.append(utterance)
                pbar.update(1)
            with open(target_file,'w') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            pbar.set_description("Saving to file {}".format(target_file))
    print("Opendialkg Preprocessing Finished!")



def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default='./original_data/opendialkg.csv',required=False, help="Path to the input file")
    # parser.add_argument("--out_file", type=str, required=False, help="Path to the output file")
    parser.add_argument("--out_file_train", type=str, default='./train_nph_data.json', required=False,
                        help="Path to the output train file")
    parser.add_argument("--out_file_dev", type=str, default='./dev_nph_data.json', required=False,
                        help="Path to the output test file")
    parser.add_argument("--out_file_test", type=str, default='./test_nph_data.json', required=False,
                        help="Path to the output test file")

    args = parser.parse_args()

    random.seed(0)

    convert(args.input_file, [args.out_file_train,args.out_file_dev,args.out_file_test])


if __name__ == "__main__":
    main()
# python convert_opendialkg.py --input_file data/opendialkg/opendialkg.csv