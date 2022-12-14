import cv2
import yaml

from application.main.onnx_model.base_model import BaseModel
from typing import Tuple

from application.main.onnx_model.util import *


class YoloOnnxModel(BaseModel):
    def __init__(self, cfg_file):
        super(YoloOnnxModel, self).__init__(cfg_file)
        self.input_nodes = ["images"]
        self.input_width, self.input_height = self.input_size

    def preprocess(self, input_img: np.ndarray):
        img = cv2.resize(input_img, dsize=self.input_size)
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)
        input_image /= 255.0
        image1 = np.transpose(input_image, (2, 0, 1))  # HWC -> CHE
        data = image1[np.newaxis, ...].astype(np.float32)  # 1 x CHW
        inputs = {self.input_nodes[0]: data}
        return inputs

    def run(self, input_img: np.ndarray):
        origin_h, origin_w, _ = input_img.shape

        inputs = self.preprocess(input_img)
        outputs = self.model.run(self.output_nodes, inputs)
        outputs = self.postprocess(outputs[0][0], origin_shape=(origin_w, origin_h))
        return outputs

    def postprocess(self, model_outputs, origin_shape: Tuple[int, int]):
        origin_w, origin_h = origin_shape
        nms_output = nms(model_outputs, self.param["conf_threshold"], self.param["iou_threshold"])

        nms_output[:, 0] *= origin_w / self.input_width
        nms_output[:, 1] *= origin_h / self.input_height
        nms_output[:, 2] *= origin_w / self.input_width
        nms_output[:, 3] *= origin_h / self.input_height
        return nms_output

    def __call__(self, img: np.ndarray, *args, **kwargs):
        return self.run(img)

    def get_iou_threshold(self):
        return self.param["iou_threshold"]

    def set_iou_threshold(self, value):
        if 0 <= value <= 1:
            self.param["iou_threshold"] = value
            return True
        return False

    def get_conf_threshold(self):
        return self.param["get_conf_threshold"]

    def set_conf_threshold(self, value):
        if 0 <= value <= 1:
            self.param["set_conf_threshold"] = value
            return True
        return False
