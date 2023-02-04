import gzip
import pickle

from datasets import load_dataset
from torch.utils.data import Dataset


class MRCDataset(Dataset):
    def __init__(self, data_type: str = "train"):

        """
        args:
            data_type
                dataset의 타입. ['train', 'validation']
        """
        assert data_type in ["train", "validation"], "지원하지 않는 datatype"
        dataset = load_dataset("klue", "mrc")[data_type]
        self.context = dataset["context"]
        self.question = dataset["question"]
        self.title = dataset["title"]
        self.answers = dataset["answers"]  # answers = {'answer_start': List[int], 'text': List[text]}

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        """
        Returns:
            (question, title, context, answer)
        """
        return {
            "question": self.question[idx],
            "title": self.title[idx],
            "context": self.context[idx],
            "answers": self.answers[idx]["text"],
        }


class ProjectDataset(Dataset):
    def __init__(self, dataset_path: str, data_type: str = "train", sep_token: str = "[SEP]", use_gzip=False):

        """
        args:
            dataset_path
                dataset_path 디렉토리.
                해당 path 안에는 train_query.pickle, valid_query.pickle이 있어야 함
            data_type
                dataset의 타입. ['train', 'valid']
            sep_token
        """
        type_list = ["train", "valid"]
        self.sep_token = sep_token
        self.context, self.question, self.answers = [], [], []
        if data_type in type_list:
            self.load_data(dataset_path, data_type, use_gzip)
        elif data_type == "all":
            for d_type in type_list:
                self.load_data(dataset_path, d_type, use_gzip)
        else:
            raise Exception("올바르지 않은 date_type")

    def load_data(self, path, data_type, use_gzip):
        pickle_path = f"{path}/{data_type}_query.pickle"
        if use_gzip:
            with gzip.open(pickle_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            with open(pickle_path, "rb") as f:
                dataset = pickle.load(f)
        self.context.extend(dataset["passage"].values.tolist())
        self.question.extend([self.sep_token.join(questions) for questions in dataset["query"].values.tolist()][:30])
        self.answers.extend(dataset["annotation"].values.tolist()[:30])

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        """
        Returns:
            (question, context, answer)
        """
        return {
            "question": self.question[idx],
            "context": self.context[idx],
            "answer": self.answers[idx]["summary1"],
        }
