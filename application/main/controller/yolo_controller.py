import logging

from fastapi import APIRouter, File, UploadFile, HTTPException
from application.main.model.yolo_model import YoloModel, YoloDetectionResult
from application.main.model.common_model import AIModelInfo
from application.main.service import yolo_service

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/", response_model=AIModelInfo)
def get_model_info():
    model_name = "Yolo V5"
    label_list = yolo_service.get_label_list()

    return AIModelInfo(**{"model_name": model_name, "label_list": label_list})


@router.post("/predict", response_model=YoloDetectionResult)
async def post_yolo_predict(file: UploadFile = File(..., content_type="image/jpeg")):
    # TODO:: Apply YoloModel
    if file.content_type != "image/jpeg":
        raise HTTPException(400, detail=f"Invalid document type Expect image/jpeg but {file.content_type}")
    output = await yolo_service.yolo_run(file)
    return YoloDetectionResult(**output)


@router.get("/param")
def get_model_param():
    return yolo_service.get_param()


@router.get("/param/iou-threshold")
def get_model_iou_threshold() -> float:
    return yolo_service.get_param_iou_threshold()


@router.put("/param/iou-threshold/{value}")
def set_model_iou_threshold(value: float) -> bool:
    return yolo_service.set_param_iou_threshold(value)


@router.get("/param/conf-threshold")
def get_model_iou_threshold() -> float:
    return yolo_service.get_param_conf_threshold()


@router.put("/param/conf-threshold/{value}")
def set_model_iou_threshold(value: float) -> bool:
    return yolo_service.set_param_conf_threshold(value)
