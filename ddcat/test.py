import torch
import torchinfo 
from model import PSPNet

model = PSPNet(layers=50, classes=21, zoom_factor=8, pretrained=False)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("./ckpt/pretrained_model/pretrain/voc2012/pspnet/no_defense/train_epoch_50.pth")
model.load_state_dict(checkpoint['state_dict'], strict=False)
print(model.state_dict().keys())
# torchinfo.summary(model, input_size=(1, 3, 224, 224))