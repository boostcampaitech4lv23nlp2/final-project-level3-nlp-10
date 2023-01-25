from typing import List


def calc_retrieval_accuracy(passages: List[tuple], answers: List[int]) -> float:
    total = len(answers)
    correct = 0
    for answer, passage in zip(answers, passages):
        if answer in passage:
            correct += 1

    return "{:.3f}".format(correct / total)
