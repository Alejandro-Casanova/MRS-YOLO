from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

# Load Config
cfg = yaml_load(check_yaml("cfg/my_config.yaml"))

# Load a pretrained model
model = YOLO('yolov8n.pt')  

results = model.train(**cfg)