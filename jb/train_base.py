from preprocessing import get_logger


class BaseTrainer:
    """

    """
    def __init__(self, epochs):
        self.epochs = epochs
        self.logger = get_logger(BaseTrainer.__name__)

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def save_model(self, fpath):
        raise NotImplementedError
