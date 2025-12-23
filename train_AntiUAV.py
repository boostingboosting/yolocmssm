import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def freeze_rgb_backbone(trainer):
    print("freeze_rgb_backbone..........................................................................")

    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("enc_rgb"):
            print("True..........................................................................")
            module.init_weights(
                pretrained="/home/zhangquan/clg/efficientvit_b1_r288.pt")
            module.eval()
            module.requires_grad = False



if __name__ == '__main__':
  
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBRGB6C-midfusion-cmssm.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
# model.load(r'yolov8n.pt') # loading pretrain weights
    # model = YOLO("/home/zhangquan/clg/YOLOv11-RGBT-master/runs/AntiUAV/AntiUAV-yolo11n-100epoch-batch32-RGBRGB6C-midfusion-cmssm-mi/weights/last.pt")
    # model = YOLO("/home/zhangquan/clg/YOLOv11-RGBT-master/runs/AntiUAV/AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain000513/weights/last.pt")
    # model.add_callback("on_train_start", freeze_rgb_backbone)
    model.train(data=R'ultralytics/cfg/datasets/AntiUAV-rgbt.yaml',
                cache=False,   
                imgsz=640,
                epochs=150,
                batch=32,
                close_mosaic=0, ###关闭mosaic
                workers=64,
                # device='4',
                device='0',
                optimizer='SGD',  # using SGD
                resume=True,
                # resume='/home/zhangquan/clg/YOLOv11-RGBT-master/runs/AntiUAV/AntiUAV-yolo11n-RGBRGB6C-midfusion20/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['infrared','visible'],
                pairs_rgb_ir=['visible','infrared'],
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/AntiUAV',
                name='AntiUAV-yolo11n-150epoch-batch64-RGBRGB6C-midfusion-cmssm-offsetgain001',
                )

        # Add a callback to put the frozen layers in eval mode to prevent BN values from changing
