import torch
import torch.nn as nn
from .Efficientvit import Encoder_Efficientvit
from .Efficientvit import Encoder_RGBT_Efficientvit
# from models.decoder.MLP import Decoder_MLP
# from models.decoder.MLP_plus import Decoder_MLP_plus
# from models.decoder.MLP_antiUAV import Detector_AntiUAV
import torch.nn.functional as F
# from proposed.fuison_strategy.base_fusion import Fusion_Module
from .fusion import Fusion_Module, RGBAdjuster
# from models.decoder.DeepLabV3 import DeepLabHeadV3Plus

class CMSSM(nn.Module):
    def __init__(self, mode='b1', inputs='rgbt', n_class=1, fusion_mode='CM-SSM', norm_fuse=nn.BatchNorm2d):
        super(CMSSM, self).__init__()
        if mode == 'b0':
            channels = [16, 32, 64, 128]
            emb_c = 128
        elif mode == 'b1':
            channels = [32, 64, 128, 256]
            emb_c = 256
        elif mode == 'b2':
            channels = [48, 96, 192, 384]
            emb_c = 256
        elif mode in ['b3', 'l1', 'l2']:
            channels = [64, 128, 256, 512]
            emb_c = 768
        self.inputs = inputs
        if inputs == 'unimodal':
            self.encoder = Encoder_Efficientvit(mode=mode)
        if inputs == 'rgbt':
            self.encoder = Encoder_RGBT_Efficientvit(mode=mode)
            self.adjuster = RGBAdjuster(channels=channels)
            self.fusion_module = Fusion_Module(fusion_mode=fusion_mode, channels=channels)
        # self.decoder = Detector_AntiUAV(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = DeepLabHeadV3Plus(in_channels=channels[-1], low_level_channels=channels[0], num_classes=12)

    def forward(self, x):
        rgb = x[0]
        t = x[1]
        # print("rgb.shape",rgb.shape)#([1, 3, 256, 256])
        # print("t.shape",t.shape)#([1, 3, 256, 256])
        if t == None:
            t = rgb
        if self.inputs == 'unimodal':
            fusions = self.encoder(rgb)
        else:
            f_rgb, f_t = self.encoder(rgb, t)
            f_rgb, f_t, txtys = self.adjuster(rgb, t)
            # print("f_rgb.shape",f_rgb[0].shape)#([1, 32, 64, 64])
            # print("f_rgb.shape",f_rgb[1].shape)#([1, 64, 32, 32])
            # print("f_rgb.shape",f_rgb[2].shape)#([1, 128, 16, 16])
            # print("f_rgb.shape",f_rgb[3].shape)#([1, 256, 8, 8])
            fusions = self.fusion_module(f_rgb, f_t)
        # print("fusion.shape",fusions[0].shape)#([1, 32, 64, 64])
        # print("fusion.shape",fusions[1].shape)#([1, 64, 32, 32])
        # print("fusion.shape",fusions[2].shape)#([1, 128, 16, 16])
        # print("fusion.shape",fusions[3].shape)#([1, 256, 8, 8])

        return fusions,txtys
        


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 480, 640).to('cuda:0')
    t = torch.rand(1, 3, 480, 640).to('cuda:0')
    model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12,).eval().to('cuda:0')
    out = model(rgb, t)
    print(out.shape)

    # from ptflops import get_model_complexity_info

    # flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    # print('Flops ' + flops)
    # print('Params ' + params)

    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    #
    # print(flop_count_table(FlopCountAnalysis(model, rgb)))
    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")
