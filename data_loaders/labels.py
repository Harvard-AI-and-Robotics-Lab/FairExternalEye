"""Shared outcome definitions used by every training pipeline."""

from collections.abc import Sequence


TASK_INDEX = {"dr": -3, "amd": -2, "glaucoma": -1}


def numeric_label(value) -> float:
    return float(value)


def is_valid_label(value, task: str) -> bool:
    value = numeric_label(value)
    if task == "dr":
        return value in {0.0, 1.0}
    if task == "amd":
        return value in {0.0, 1.0, 2.0, 3.0, 255.0}
    if task == "glaucoma":
        return value in {-1.0, 0.0, 1.0}
    raise ValueError(f"Unsupported task: {task}")


def binary_label(value, task: str) -> int:
    value = numeric_label(value)
    if not is_valid_label(value, task):
        raise ValueError(f"Invalid {task} label: {value}")
    if task == "dr":
        return int(value == 1.0)
    if task == "amd":
        return int(value == 3.0)
    return int(value == 1.0)


def filter_valid_samples(samples: Sequence, task: str) -> list:
    task_index = TASK_INDEX[task]
    return [sample for sample in samples if is_valid_label(sample[task_index], task)]
