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
    def __init__(self, dataset_path: str, data_type: str = "train"):

        """
        args:
            dataset_path
                dataset_path 디렉토리.
                해당 path 안에는 train_query.pickle, valid_query.pickle이 있어야 함
            data_type
                dataset의 타입. ['train', 'valid']
        """
        assert data_type in ["train", "valid"], "지원하지 않는 datatype"
        pickle_path = f"{dataset_path}/{data_type}_query.pickle"
        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)
        self.context = dataset["passage"].values.tolist()
        self.question = dataset["query"].values.tolist()
        self.answers = dataset[
            "annotation"
        ].values.tolist()  # answers = {'answer_start': List[int], 'text': List[text]}

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
