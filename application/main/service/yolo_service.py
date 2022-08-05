# Real Logic..
import logging

import cv2
import numpy as np
import numpy as np
from fastapi import UploadFile, HTTPException

from application.main.model.yolo_model import YoloModel
from application.main.onnx_model.yolo_model import YoloOnnxModel, get_labels

onnx_model = YoloOnnxModel(r"static\onnx_file\yolov5s.onnx")

logger = logging.getLogger(__name__)

async def yolo_run(file: UploadFile):
    logger.info(f"filename : {file.filename}")
    contents = await file.read()
    binarydata = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(binarydata, cv2.IMREAD_COLOR)

    if img is None:
        logger.error(f"image read fail :{file.filename}")
        raise HTTPException(400, "image read failed")

    iou_thd = 0.3  # default iou_thd
    conf_thd = 0.3  # default conf_thd

    output = onnx_model(img, conf_thd=conf_thd, iou_thd=iou_thd)

    to_result = dict()
    to_result["number_objects"] = len(output)
    objects = list()
    for minx, miny, maxx, maxy, conf, idx in output:
        obj = {"label": get_labels(int(idx)),
               "bbox": [minx, miny, maxx, maxy],
               "conf": conf}
        objects.append(obj)
    to_result["objects"] = objects

    return to_result
