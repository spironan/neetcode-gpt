import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        words = [word for sentence in positive + negative for word in sentence.split()]
        vocab = sorted(set(words))
        lookup = {word: index + 1 for index, word in enumerate(vocab)}
        
        output : TensorType[float] = []

        combine = positive + negative
        for sentence in combine:
            row = [lookup[word] for word in sentence.split()]
            output.append(torch.tensor(row))
        
        # finally we pad shorter sequences with 0s
        output = nn.utils.rnn.pad_sequence(output, batch_first=True)

        return output 

        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        pass
