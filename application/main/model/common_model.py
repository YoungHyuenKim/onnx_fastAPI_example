from pydantic import BaseModel
from typing import List


class AIModelInfo(BaseModel):
    model_name: str
    label_list: List[str]
