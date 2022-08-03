import abc
import json
import os

import numpy as np
import onnxruntime

from abc import ABC
from typing import Tuple


class BaseModel(ABC):

    def __init__(self, onnx_file: str, input_size: Tuple[int, int]):
        self.device = onnxruntime.get_device()
        self.provider = "CUDAExecutionProvider" if self.device == "GPU" else "CPUExecutionProvider"
        assert os.path.exists(onnx_file), f"Not Exists onnx model file [{onnx_file}]"
        self.model = onnxruntime.InferenceSession(onnx_file, providers=[self.provider])

        self.input_size = input_size
        self.input_nodes = None
        self.output_nodes = None

        self.last_input_img = None

    @abc.abstractmethod
    def preprocess(self, input_img: np.ndarray):
        pass

    @abc.abstractmethod
    def run(self, input_img: np.ndarray):
        pass

    @abc.abstractmethod
    def __call__(self, img: np.ndarray, *args, **kwargs):
        pass
