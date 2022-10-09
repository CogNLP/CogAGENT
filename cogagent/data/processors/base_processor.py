from ..datable import DataTable

class BaseProcessor:
    def __init__(self,debug=False):
        self.debug = debug

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

    def debug_process(self, data):
        if self.debug and len(data) >= 100:
            debug_data = DataTable()
            for header in data.headers:
                debug_data[header] = data[header][:100]
            return debug_data
        return data
