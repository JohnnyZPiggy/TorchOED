import torch


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")
    return value


def floating(dtype):
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("supported floating dtypes: torch.float32, torch.float64")


def budget(dimensions, max_elements):
    integer(max_elements, "max_elements")
    total = 1
    for size in dimensions:
        if size > min(max_elements, 2**63 - 1):
            raise ValueError("dimension exceeds output element budget/int64 limit")
        total *= size
        if total > min(max_elements, 2**63 - 1):
            raise ValueError("output exceeds element budget/int64 limit")
