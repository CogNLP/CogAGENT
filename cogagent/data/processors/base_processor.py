class BaseProcessor:
    def __init__(self):
        pass

    def _process(self, data):
        pass

    def process_train(self, data):
        pass

    def process_dev(self, data):
        pass

    def process_test(self, data):
        pass

    def _collate(self, batch):
        pass

    def train_collate(self, batch):
        pass

    def dev_collate(self, batch):
        pass

    def test_collate(self, batch):
        pass

    def get_addition(self):
        pass