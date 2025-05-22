from io import BytesIO
from typing import Optional
from pydantic import BaseModel
from typing import List, Optional, Dict
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from loguru import logger

from .infer import InferClient
from .infer.error import LoadError, FileSizeMismatchError, DownloadError
from .settings import InferSettingCurrent
import base64

app = FastAPI()


def verify_token(token):
    # TODO: Implement your token verification logic here
    return True


INFER_APP = InferClient(
    model_name=InferSettingCurrent.wd_model_name,
    model_dir=InferSettingCurrent.wd_model_dir,
    skip_auto_download=InferSettingCurrent.skip_auto_download,
)
logger.info(f"Infer app init success, model_path: {INFER_APP.model_path}")
class InferenceRequest(BaseModel):
    id: str
    images: List[str]
    token: Optional[str] = None
    general_threshold: Optional[float] = 0.6
    character_threshold: Optional[float] = 0.85
    general_mcut_enabled: Optional[bool] = False
    character_mcut_enabled: Optional[bool] = False


def process_base64(image_str: str) -> bytes:
    """处理可能包含前缀的base64字符串"""
    if ',' in image_str:
        # 分割数据URI前缀和实际数据
        header, data = image_str.split(',', 1)
        # 可选：验证header格式（例如是否包含base64）
        if 'base64' not in header.lower():
            raise ValueError("Invalid base64 header")
    else:
        data = image_str
    
    # 补全base64填充
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Base64解码失败: {str(e)}")



@app.post("/label")
async def label_endpoint(request: InferenceRequest):
    """处理JSON格式的图片标注请求"""
    # Token验证
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if not request.images:
        raise HTTPException(status_code=400, detail="No images provided")

    try:
        # 处理第一个图像（假设单图处理）
        image_bytes = process_base64(request.images[0])
        image = Image.open(BytesIO(image_bytes))

        # 调用推理引擎
        (
            _,
            _,
            character_res,
            general_res,
        ) = await INFER_APP.infer(
            image=image,
            general_threshold=request.general_threshold,
            character_threshold=request.character_threshold,
            general_mcut_enabled=request.general_mcut_enabled,
            character_mcut_enabled=request.character_mcut_enabled,
        )

        # 构建标签结果
        labels = []
        
        # 添加通用标签
        for tag, confidence in general_res.items():
            labels.append({
                "name": tag,
                "source": "general",
                "confidence": float(confidence),
                "priority": 0
            })
        
        # 添加角色标签（如果存在）
        if character_res:
            for char, confidence in character_res.items():
                labels.append({
                    "name": char,
                    "source": "character",
                    "confidence": float(confidence),
                    "priority": 1
                })

        # 按置信度降序排序
        sorted_labels = sorted(labels, key=lambda x: x["confidence"], reverse=True)

        return {
            "id": request.id,
            "code": 200,
            "model": {
                "type": "labels",
                "name": "vit"  # 根据实际模型名称修改
            },
            "result": {
                "labels": sorted_labels
            }
        }

    except (LoadError, DownloadError, FileSizeMismatchError) as e:
        logger.exception(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")