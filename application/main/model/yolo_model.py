from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import cv2
import numpy as np


class YoloModel(BaseModel):
    conf_thd: Optional[float] = 0.3
    iou_thd: Optional[float] = 0.3


class DetectedObject(BaseModel):
    label: str
    bbox: List[float]
    conf: float


class YoloDetectionResult(BaseModel):
    number_objects: int
    objects: List[DetectedObject]
