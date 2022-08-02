from typing import List, Optional
from pydantic import BaseModel

import cv2
import numpy as np


class YoloModel(BaseModel):
    bytes_img: bytes
    conf_thd: Optional[float] = None
    iou_thd: Optional[float] = None

    def bytes2img(self) -> np.ndarray:
        nparr = np.frombuffer(self.bytes_img, np.uint8)
        return cv2.imdecode(nparr)


class DetectedObject(BaseModel):
    label: str
    bbox: List[float]
    conf: float


class YoloRedictionResult(BaseModel):
    number_objects: int
    objects: List[DetectedObject]

