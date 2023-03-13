"""Inference Script for Simple SegNet"""
import cv2
import torchvision.transforms as tf
import torch
import torchvision.models.segmentation
import matplotlib.pyplot as plt

model_path = "model/model_10.pth"
img_path ="D:\Pytorch-UNet\data\imgs\Video1_frame000090.png"
width = height = 1024
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Net= torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
Net=Net.to(device)
Net.load_state_dict(torch.load(model_path))
Net.eval()
Img=cv2.imread(img_path)
height_orig, width_orig = Img.shape[:2]
plt.imshow(Img[:,:,::-1])
Img=transformImg(Img)
Img=torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    pred = Net(Img)['out']
Prd=tf.Resize((height_orig, width_orig))(pred[0])
seg = torch.argmax(Prd, 0).cpu().detach().numpy()
plt.imshow(seg)
plt.show()