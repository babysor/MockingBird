from abc import ABC
from abc import abstractmethod

import torch

class AbsMelDecoder(torch.nn.Module, ABC):
    """The abstract PPG-based voice conversion class
    This "model" is one of mediator objects for "Task" class.

    """

    @abstractmethod
    def forward(
        self, 
        bottle_neck_features: torch.Tensor,
        feature_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        logf0_uv: torch.Tensor = None,
        spembs: torch.Tensor = None,
        styleembs: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
