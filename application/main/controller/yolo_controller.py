import logging

from fastapi import APIRouter, File, UploadFile, HTTPException
from application.main.model.yolo_model import YoloModel, YoloDetectionResult
from application.main.service.yolo_service import yolo_run

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/predict", response_model=YoloDetectionResult)
async def post_yolo_predict(file: UploadFile = File(..., content_type="image/jpeg")):
    # TODO:: Apply YoloModel
    if file.content_type != "image/jpeg":
        raise HTTPException(400, detail=f"Invalid document type Expect image/jpeg but {file.content_type}")
    output = await yolo_run(file)
    return YoloDetectionResult(**output)


@router.get("/predict")
def health_check():
    # TODO:: Health Check YoloModel
    return "Health Check"
