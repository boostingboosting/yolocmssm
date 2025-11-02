import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
  
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBRGB6C-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
# model.load(r'yolov8n.pt') # loading pretrain weights

    model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=64,
                close_mosaic=0,
                workers=64,
                device='1,2,4,5',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/AntiUAV',
                name='AntiUAV-yolo11n-RGBRGB6C-midfusion',
                )