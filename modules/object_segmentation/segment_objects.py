import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

# Load configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "2115979251_8281e3fe36_b.jpg")

# Load image
im = cv2.imread(img_path)

# Run prediction
outputs = predictor(im)

# Access segmentation masks
masks = outputs["instances"].pred_masks.cpu().numpy()
classes = outputs["instances"].pred_classes.cpu().numpy()

# Process and visualize masks (example)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Segmentation", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
