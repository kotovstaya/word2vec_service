import re
import numpy as np
import typing as tp
import logging
import os
import pandas as pd

import pygments
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_for_filename


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


class FormatterProxy(Formatter):
    name = "Proxy"
    aliases = ["proxy"]
    filenames = []

    def __init__(self, **options):
        super(FormatterProxy, self).__init__(**options)
        self.callback = options["callback"]

    def format(self, tokensource, outfile):
        self.callback(tokensource)


class FileParser:

    def __init__(self, name_breakup_re_str: tp.Optional[str] = None):
        self.NAME_BREAKUP_RE = re.compile(name_breakup_re_str if name_breakup_re_str else r"[^a-zA-Z]+")
        self.names = []

    def extract_names(self, token):
        token = token.strip()
        prev_p = [""]

        def ret(name):
            r = name.lower()
            if len(name) >= 3:
                yield r
                if prev_p[0]:
                    yield prev_p[0] + r
                    prev_p[0] = ""
            else:
                prev_p[0] = r

        for part in self.NAME_BREAKUP_RE.split(token):
            if not part:
                continue
            prev = part[0]
            pos = 0
            for i in range(1, len(part)):
                this = part[i]
                if prev.islower() and this.isupper():
                    yield from ret(part[pos:i])
                    pos = i
                elif prev.isupper() and this.islower():
                    if 0 < i - 1 - pos <= 3:
                        yield from ret(part[pos:i - 1])
                        pos = i - 1
                    elif i - 1 > pos:
                        yield from ret(part[pos:i])
                        pos = i
                prev = this
            last = part[pos:]
            if last:
                yield from ret(last)

    def process_tokens(self, tokens) -> None:
        self.names = np.concatenate(
            [list(self.extract_names(value)) for _type, value in tokens if _type[0] == "Name"]
        )

    def parse(self, txt: str) -> tp.List[str]:
        lexer = get_lexer_for_filename("foo.py", txt)
        pygments.highlight(txt, lexer, FormatterProxy(callback=self.process_tokens))
        return self.names


class FilesMerger:
    def __init__(self, data_directory: str, raw_dataset_fpath: str, files_fpath: str):
        self.file_descriptor = open(os.path.join(data_directory, raw_dataset_fpath), "a")
        self.data_directory = data_directory
        self.files_fpath = files_fpath
        self.FILES = set()
        self.BAD_FILES = set()

    def merge(self, entities: np.array) -> None:
        self.file_descriptor.write(f"{' '.join(entities)}\n")

    def close(self) -> None:
        self.file_descriptor.close()
        files = pd.DataFrame(data={"file": [file for file in self.FILES if file.endswith('.py')]})
        files.to_csv(f"{self.data_directory}/{self.files_fpath}", index=False, sep=";")


class LocalFileReader:
    @classmethod
    def read(cls, path: str) -> str:
        with open(path, "r") as f:
            return "".join(f.readlines())


class BaseRepositoryExtractor:
    def __init__(self, file_merger):
        self.NAMES = []
        self.file_merger = file_merger

    def extract_recursively(self):
        raise NotImplementedError

    def _get_text(self, filepath):
        raise NotImplementedError


class LocalRepositoryExtractor(BaseRepositoryExtractor):
    def __init__(self, repo_path: str, file_merger: FilesMerger):
        super().__init__(file_merger)
        self.repo_path = repo_path
        self.logger = get_logger(LocalRepositoryExtractor.__name__)

    def _get_text(self, filepath):
        txt = LocalFileReader().read(filepath)
        return txt

    def extract_recursively(self):
        def _inner_rec_func(base_path: str):
            for file in os.listdir(base_path):
                path = os.path.join(base_path, file)
                if os.path.isdir(path):
                    self.file_merger.FILES.add(file)
                    _inner_rec_func(path)
                else:
                    if file.endswith(".py"):
                        self.logger.warning(f"{file} is python file")
                        try:
                            txt = self._get_text(path)
                            fp = FileParser()
                            entities = fp.parse(txt)
                            self.file_merger.merge(entities)
                            self.file_merger.FILES.add(path)
                        except Exception as ex:
                            self.file_merger.BAD_FILES.add(path)

        _inner_rec_func(self.repo_path)
        self.file_merger.close()


if __name__ == "__main__":

    data_folder = "./../data"
    repository_path = "./../git_clone/pandas"

    fm = FilesMerger(data_directory=data_folder,
                     raw_dataset_fpath="raw_dataset.txt",
                     files_fpath="seen_files.csv")
    repo_ext = LocalRepositoryExtractor(repository_path, fm)
    repo_ext.extract_recursively()