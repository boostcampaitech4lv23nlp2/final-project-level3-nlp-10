from typing import List


def calc_retrieval_accuracy(passages: List[tuple], answers: List[int]) -> float:
    total = len(answers)
    correct = 0
    for answer, passage in zip(answers, passages):
        if answer in passage:
            correct += 1

    return "{:.3f}".format(correct / total)


top3_passages = [(1, 2, 3), (3, 4, 1), (1, 11, 13)]
answers = [1, 5, 2]
print(calc_retrieval_accuracy(top3_passages, answers))
