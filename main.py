import base64
from io import BytesIO

from PIL import Image
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("./runs/detect/train2/weights/best.pt")


# 定义一个请求体模型
class Item(BaseModel):
    bg: str


def base64_to_image(base64_string, output_path=None):
    # 移除可能存在的头部信息（例如：data:image/png;base64,）
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # 解码 Base64 字符串
    image_data = base64.b64decode(base64_string)

    # 将解码后的数据转换为 PIL Image 对象
    image = Image.open(BytesIO(image_data))
    # 保存图片
    # image.save(output_path)
    return image


@app.get("/")
def jy_slide_page():
    return FileResponse(path='index.html', media_type='text/html')


@app.post("/jy_slide")
def jy_slide(file: UploadFile = File(...)):
    rs = []
    # im1 = base64_to_image(base64_string=item.bg)
    results = model.predict(source=Image.open(file.file))  # 保存绘制后的图像
    for result in results:
        boxes = result.boxes  # 获取检测到的所有边界框
        for box in boxes:
            # 每个box包含坐标和其他信息
            x_min, y_min, x_max, y_max = box.xyxy[0]  # 获取左上角和右下角坐标
            print(f"Detected object at: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            rs.append({"x_min": int(x_min), "x_max": int(x_max), "y_min": int(y_min), "y_max": int(y_max)})
    return {"status":0,"msg":"成功","data":rs}
