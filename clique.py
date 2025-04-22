from typing import List
from abc import ABC, abstractmethod
import torch

class Clique(ABC):
    """A general interface for representing subsets (or cliques) defined by some common rule e.g. all words that end in blank"""

    @abstractmethod
    def encode_sample(self, sample) -> torch.Tensor:
        """Return the sample encoded as a PyTorch tensor"""
        pass

    @abstractmethod
    def generate_example(self):
        """Returns a random example based on the clique's rules"""
        pass

    @abstractmethod
    def generate_sample(self):
        """Returns a random sample from the clique's superset"""
        pass

    @abstractmethod
    def __contains__(self, sample) -> bool:
        """Return whether the input `sample` is included in this clique"""
        pass

    def generate_counterexample(self):
        raise NotImplementedError

    def generate_examples(self, n: int) -> List:
        return [self.generate_example() for _ in range(n)]

    def generate_samples(self, n: int) -> List:
        return [self.generate_sample() for _ in range(n)]

    def generate_counterexamples(self, n: int) -> List:
        """Use the custom helper function if available; otherwise fall back to rejection sampling"""
        try:
            return [self.generate_counterexample() for _ in range(n)]
        except NotImplementedError:
            lst = []
            while len(lst) < n:
                sample = self.generate_sample()
                if sample not in self: # add elements that aren't in the clique
                    lst.append(sample)
            return lst

    def encode_samples(self, samples: List) -> torch.Tensor:
        """Encode a list of samples as a Tensor of size n x d where n is the length of the list"""
        return torch.stack([self.encode_sample(sample) for sample in samples])
