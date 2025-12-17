import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(R'LLVIP/LLVIP-yolov8-RGBT-midfusion/weights/best.pt')
    model = YOLO("/home/zhangquan/clg/YOLOv11-RGBT-master/runs/AntiUAV/AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi-flow/weights/best.pt")
    model.val(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
              split='val',
              imgsz=640,
              batch=32,
              device='0',
              use_simotm="RGBRGB6C",
              channels=6,
              # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/AntiUAV',
              name='AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi-flow',
              )