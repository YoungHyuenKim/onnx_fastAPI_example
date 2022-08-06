import logging

import cv2
import numpy as np
import numpy as np
from fastapi import UploadFile, HTTPException

from application.main.model.yolo_model import YoloModel
from application.main.onnx_model.yolo_model import YoloOnnxModel

onnx_model = YoloOnnxModel("main/onnx_model/config/yolo.yaml")

logger = logging.getLogger(__name__)


async def yolo_run(file: UploadFile):
    logger.info(f"filename : {file.filename}")
    contents = await file.read()
    binarydata = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(binarydata, cv2.IMREAD_COLOR)

    if img is None:
        logger.error(f"image read fail :{file.filename}")
        raise HTTPException(400, "image read failed")

    output = onnx_model(img)

    to_result = dict()
    to_result["number_objects"] = len(output)
    objects = list()
    for minx, miny, maxx, maxy, conf, idx in output:
        obj = {"label": onnx_model.get_label(int(idx)),
               "bbox": [minx, miny, maxx, maxy],
               "conf": conf}
        objects.append(obj)
    to_result["objects"] = objects

    return to_result


def get_param():
    return onnx_model.get_param()


def get_param_iou_threshold():
    return onnx_model.get_iou_threshold()


def set_param_iou_threshold(value):
    return onnx_model.set_iou_threshold(value)


def get_param_conf_threshold():
    return onnx_model.get_conf_threshold()


def set_param_conf_threshold(value):
    return onnx_model.set_conf_threshold(value)


def get_label_list():
    return onnx_model.get_full_labels()
