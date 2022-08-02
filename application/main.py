from typing import Union
import logging

import uvicorn
from fastapi import FastAPI
from application.main.controller import *

logger = logging.getLogger(__name__)


def include_router(app: FastAPI):
    app.include_router(yolo_controller.router, prefix="/yolo")


def start_application():
    app = FastAPI()
    include_router(app)
    return app


app = start_application()

if __name__ == '__main__':
    uvicorn.run(app)
