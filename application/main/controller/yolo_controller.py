import logging

from fastapi import APIRouter, Depends
from application.main.model.yolo_model import YoloModel, YoloRedictionResult
from application.main.service.yolo_service import yolo_run

router = APIRouter()


@router.post("/predict", response_model=YoloRedictionResult)
def post_yolo_predict(data: YoloModel):
    output = yolo_run(data)
    return YoloRedictionResult(**output)


@router.get("/predict")
def health_check():
    return "Health Check"

