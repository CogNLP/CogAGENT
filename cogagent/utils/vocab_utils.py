import pickle
class Vocabulary():
    def __init__(self):
        self.label_set = set()
        self.defined_label2id_dict = dict()
        self.label2id_dict = {}
        self.id2label_dict = {}

    def __len__(self):
        return len(self.label_set)

    def add(self, label):
        self.label_set.add(label)

    def add_sequence(self, labels):
        for label in labels:
            self.add(label)

    def add_dict(self, defined_label2id_dict):
        for label, id in defined_label2id_dict.items():
            if label not in self.defined_label2id_dict:
                self.defined_label2id_dict[label] = id
                self.label_set.add(label)

    def create(self):
        label_list = list(self.label_set)
        label_list.sort()
        defined_label_set = set()
        for label, id in self.defined_label2id_dict.items():
            self.label2id_dict[label] = id
            self.id2label_dict[id] = label
            defined_label_set.add(label)
            if id >= len(label_list) or id < 0:
                raise IndexError("Defined dict id must smaller than label num and bigger than 0.")
        inserting_label_set = self.label_set - defined_label_set
        inserting_label_list = list(inserting_label_set)
        inserting_label_list.sort()
        inserting_index = 0
        for id in range(len(self.label_set)):
            if id not in self.id2label_dict.keys():
                self.label2id_dict[inserting_label_list[inserting_index]] = id
                self.id2label_dict[id] = inserting_label_list[inserting_index]
                inserting_index += 1
        self.label2id_dict = dict(sorted(self.label2id_dict.items(), key=lambda x: x[1]))
        self.id2label_dict = dict(sorted(self.id2label_dict.items(), key=lambda x: x[0]))

    def label2id(self, word):
        return self.label2id_dict[word]

    def id2label(self, id):
        return self.id2label_dict[id]

    def labels2ids(self, words):
        return [self.label2id(word) for word in words]

    def ids2labels(self, ids):
        return [self.id2label(id) for id in ids]

    def get_label2id_dict(self):
        return self.label2id_dict

    def get_id2label_dict(self):
        return self.id2label_dict

    def save(self,file):
        data = [self.label_set,self.defined_label2id_dict,self.label2id_dict,self.id2label_dict]
        with open(file,"wb") as f:
            pickle.dump(data,f)

    def load(self,file):
        with open(file,"rb") as f:
            data = pickle.load(f)
        self.label_set,self.defined_label2id_dict,self.label2id_dict,self.id2label_dict = data


if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.add("C")
    vocab.add("A")
    vocab.add("B")
    vocab.add("A")
    vocab.add_sequence(["C", "B", "D"])
    vocab.add_dict({"<pad>": 0})
    vocab.add_dict({"A": 2})
    vocab.create()

    file_name = "vocabulary.pkl"
    vocab.save(file_name)
    another_vocab = Vocabulary()
    another_vocab.load(file_name)

    print(vocab.label2id("A"))
    print(vocab.id2label(0))
    print(vocab.labels2ids(["A", "B"]))
    print(vocab.ids2labels([0, 1]))
    print(vocab.get_label2id_dict())
    print(vocab.get_id2label_dict())
    print(len(vocab))
    print("end")