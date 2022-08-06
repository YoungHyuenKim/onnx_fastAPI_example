import abc
import os
import yaml
import numpy as np
import onnxruntime

from abc import ABC

from .config import util as config_util


class BaseModel(ABC):

    def __init__(self, cfg_file: str):
        self.cfg_file = cfg_file
        cfg_yaml = config_util.load_config(self.cfg_file)
        self.model_file = cfg_yaml["onnx_file"]
        self.input_size = (cfg_yaml["input_height"], cfg_yaml["input_height"])
        self.input_nodes = cfg_yaml["input_nodes"]
        self.output_nodes = cfg_yaml["output_nodes"]
        self.param = cfg_yaml["param"]
        self.label_file = cfg_yaml["label_file"]
        with open(self.label_file, "r") as f:
            self.labels = yaml.load(f, yaml.SafeLoader)
        assert os.path.exists(self.model_file), f"Not Exists onnx model file [{self.model_file}]"

        self.device = onnxruntime.get_device()
        self.provider = "CUDAExecutionProvider" if self.device == "GPU" else "CPUExecutionProvider"
        self.model = onnxruntime.InferenceSession(self.model_file, providers=[self.provider])

    @abc.abstractmethod
    def preprocess(self, input_img: np.ndarray):
        pass

    @abc.abstractmethod
    def run(self, input_img: np.ndarray):
        pass

    @abc.abstractmethod
    def __call__(self, img: np.ndarray, *args, **kwargs):
        pass

    def get_full_labels(self):
        return self.labels

    def get_label(self, idx: int):
        return self.labels[idx]

    def get_param(self):
        return self.param

    def set_param(self, key, value):
        if key in self.param:
            if type(self.param[key]) == type(value):
                self.param[key] = value
            return True
        return False

    def save_config(self):
        config = dict()

        config["_version"] = 1
        config["onnx_file"] = self.model_file
        config["input_width"] = self.input_size[0]
        config["input_height"] = self.input_size[1]

        config["input_nodes"] = self.input_nodes
        config["output_nodes"] = self.output_nodes

        config["param"] = self.param
        config["label_file"] = self.label_file

        if config_util.valid_config(config):
            config_util.save_config(config, self.cfg_file)
