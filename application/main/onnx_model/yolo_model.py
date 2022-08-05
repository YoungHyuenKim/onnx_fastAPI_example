import onnxruntime
import numpy as np
import cv2
from application.main.onnx_model.base_model import BaseModel
from typing import Tuple

from application.main.onnx_model.util import *

yolo_onnx_file = "application\static\onnx_file\yolov5s.onnx"

yolo_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_labels(idx: int):
    return yolo_labels[idx]


class YoloOnnxModel(BaseModel):
    def __init__(self, model_file, input_size=(640, 640)):
        super(YoloOnnxModel, self).__init__(model_file, input_size)
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

    def run(self, input_img: np.ndarray, *args, **kwargs):
        origin_h, origin_w, _ = input_img.shape

        inputs = self.preprocess(input_img)
        outputs = self.model.run(self.output_nodes, inputs)  # outputs: 1 x N x 85(xyxy conf + class conf(80))
        outputs = self.postprocess(outputs[0][0], origin_shape=(origin_w, origin_h), conf_thd=kwargs["conf_thd"],
                                   iou_thd=kwargs["iou_thd"])
        return outputs

    def postprocess(self, model_outputs, origin_shape: Tuple[int, int], conf_thd, iou_thd):
        origin_w, origin_h = origin_shape
        nms_output = nms(model_outputs, conf_thd, iou_thd)

        nms_output[:, 0] *= origin_w / self.input_width
        nms_output[:, 1] *= origin_h / self.input_height
        nms_output[:, 2] *= origin_w / self.input_width
        nms_output[:, 3] *= origin_h / self.input_height
        return nms_output

    def __call__(self, img: np.ndarray, conf_thd=0.3, iou_thd=0.3, *args, **kwargs):
        return self.run(img, conf_thd=conf_thd, iou_thd=iou_thd)
