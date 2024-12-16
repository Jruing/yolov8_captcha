# yolov8_captcha
> 采集某验滑块验证码，使用YOLO进行验证码缺口位置识别

```python
main.py # 提供页面上传返回缺口坐标信息
script/train.py # 训练脚本
script/yolov8n.pt # 模型文件
script/inference.py # 利用训练好的模型进行缺口位置推理
script/change_label_info.py # 修改标签信息
script/validation.py # 验证
```
