from preprocessing import get_logger


class BaseTrainer:
    """
    Base class for a model training.
    Args:
        :params epochs: how long will it take to train one model
    Methods:
        _init_model  - how we should initialize our model
        train_model - implement this method in children class if you want train a model
        save_model - how and where should we save our model
    """
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.logger = get_logger(BaseTrainer.__name__)

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def save_model(self, fpath):
        raise NotImplementedError
