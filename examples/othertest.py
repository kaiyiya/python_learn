import torch
import numpy as np


def main():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2], [4, 5], [6, 7]])
    print(t1.sum(dim=1))


if __name__ == '__main__':
    main()
