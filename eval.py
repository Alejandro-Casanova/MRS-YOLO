from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

# model = YOLO("./best/bestC3.pt")  # Load a pretrained model
# results = model.val(data="cfg/datasets/data.yaml", imgsz=640)
# print(results.box.map)

# Load Config
cfg = yaml_load(check_yaml("cfg/my_config.yaml"))
model = YOLO('yolov8n.pt')  # Load a pretrained model
results = model.train(**cfg)