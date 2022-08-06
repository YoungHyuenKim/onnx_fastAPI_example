import yaml
import logging

logger = logging.getLogger(__name__)


def get_dafault_onnx_config():
    config_data = dict()

    config_data["_version"] = 1
    config_data["onnx_file"] = ""
    config_data["input_width"] = 0
    config_data["input_height"] = 0

    config_data["input_nodes"] = None
    config_data["output_nodes"] = None

    param = dict()

    config_data["param"] = param
    config_data["label_file"] = ""

    return config_data


def valid_config(cfg: dict):
    NECESSARY_KEY = ["_version", "onnx_file", "input_width", "input_height", "input_nodes", "output_nodes", "param",
                     "label_file"]

    for key in NECESSARY_KEY:
        if key not in cfg:
            return False


def save_config(cfg: dict, file: str):
    if not valid_config(cfg):
        logger.error(f"config file is invalid. Do not Save to {file}")
        return

    with open(file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)


def load_config(file):
    with open(file, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return cfg_yaml
