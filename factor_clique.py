import random
import torch

from clique import Clique

MAX_INT = 2 ** 31 - 1

class FactorClique(Clique):
    def __init__(self, factor, max_val=MAX_INT):
        assert factor > 10
        assert (max_val / factor) > 10

        self.factor = factor
        self.max_val = max_val

    def generate_example(self) -> int:
        return random.randint(1, self.max_val // self.factor) * self.factor

    def generate_counterexample(self) -> int:
        # subtract one to ensure val < max_val
        val = random.randint(1, self.max_val // self.factor - 1) * self.factor
        return val + random.randint(1, self.factor - 1)

    def generate_sample(self) -> int:
        return random.randint(1, self.max_val)

    def encode_sample(self, n) -> torch.Tensor:
        assert isinstance(n, int) and n > 0

        e = torch.zeros(32)
        binary = bin(n)[2:]
        for i in range(1, len(binary) + 1):
            e[-i] = int(binary[-i])
        return e

    def __contains__(self, sample: int) -> bool:
        assert isinstance(sample, int) and sample > 0, f"Invalid sample: {sample}"
        return sample % self.factor == 0