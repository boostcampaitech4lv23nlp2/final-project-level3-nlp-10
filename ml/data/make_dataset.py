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
            return (self.question[idx], self.title[idx], self.context[idx], self.answers[idx]["text"])
