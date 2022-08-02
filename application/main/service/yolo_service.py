# Real Logic..
import logging

import numpy as np
from application.main.model.yolo_model import YoloModel

logger = logging.getLogger(__name__)


def yolo_run(data: YoloModel):
    # img = data.bytes2img()
    iou_thd = data.iou_thd if data.iou_thd else 0.3  # default iou_thd
    conf_thd = data.conf_thd if data.conf_thd else 0.3  # default conf_thd
    # logger.info(f"img_shape : {img.shape}, iou_thd : {data.iou_thd}, conf_thd : {data.conf_thd}")
    logger.info(f"img_shape : not implements, iou_thd : {data.iou_thd}, conf_thd : {data.conf_thd}")
    # model run
    # nms algorithm.

    output = np.array([[100, 100, 200, 200, 1, 0.6],
                       [200, 200, 400, 400, 2, 0.7]])

    to_result = dict()
    to_result["number_objects"] = len(output)
    objects = list()
    for minx, miny, maxx, maxy, idx, conf in output:
        obj = {"label": idx2label(idx),
               "bbox": [minx, miny, maxx, maxy],
               "conf": conf}
        objects.append(obj)
    to_result["objects"] = objects

    return to_result


def idx2label(idx: int):
    return f"label{idx}"
