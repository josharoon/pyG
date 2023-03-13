import os,cv2
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

learning_rate = 0.001
batch_size = 8
width = height = 256
DATA_DIR = "D:\Pytorch-UNet\data"
TRAIN_FILE = "train.lst"
ROOT="D:\pyG"

# convert .lst to list of images and masks
def readList():
    img_lst=[]
    mask_lst=[]
    with open(os.path.join(DATA_DIR, TRAIN_FILE), 'r') as f:
        for line in f:
            img, mask = line.strip().split()
            img_lst.append(img)
            mask_lst.append(mask)
    return img_lst, mask_lst


ListImg,ListMask=readList()

print("traing dataset size: ", len(ListImg))
print("mask dataset size: ", len(ListMask))

#define transforms
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor()])

def readRandomImage():
    index = np.random.randint(0, len(ListMask))
    img = cv2.imread(os.path.join(DATA_DIR, ListImg[index]))
    # read in mask as 1 channel image
    mask = cv2.imread(os.path.join(DATA_DIR, ListMask[index]), 0)
    # convert mask to float
    mask = mask.astype(np.float32)

    # copy mask to 3 channels
    #mask = np.concatenate((mask, mask, mask), axis=2)
    imgTensor = transformImg(img)
    maskTensor = transformAnn(mask)
    return imgTensor, maskTensor

def loadBatch():
    images = torch.zeros([batch_size, 3, height, width])
    ann = torch.zeros([batch_size, height, width])

    for i in range(batch_size):
        images[i], ann[i] = readRandomImage()

    return images, ann

#load model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

#train
for epoch in range(100):
    for i in range(1000):
        imgBatch, maskBatch = loadBatch()
        images= torch.autograd.Variable(imgBatch, requires_grad=False).to(device)
        labels= torch.autograd.Variable(maskBatch, requires_grad=False).to(device)
        pred = model(images)['out']
        loss = criterion(pred, labels.long())
        loss.backward()
        optimizer.step()
        # print statistics every 100 steps
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

    #save model
    torch.save(model.state_dict(), os.path.join(ROOT,"model", "model_" + str(epoch) + ".pth"))
