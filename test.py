import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from ViT import VIT
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
file_path = 'D:/lnx/code/dataset/flowers recognition/test'
weight_path = 'D:/lnx/code/ViT save weights/epoch9_valacc100%_VIT.pth'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

data_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

data_sets = datasets.ImageFolder(file_path, transform=data_transformer)
dataloader = DataLoader(data_sets, batch_size=batch_size, shuffle=True)

test_num = len(data_sets)
class_name = dict((v, k) for k, v in data_sets.class_to_idx.items())

def img_show(imgs, labels, cls, scores):
    frames = imgs[:8]
    true_labels = labels[:8]
    pred_labels = cls[:8]
    pred_scores = scores[:8]

    frames = frames.numpy()
    frames = np.transpose(frames, [0, 2, 3, 1])

    mean = [0.485, 0.456, 0.406]  # 均值
    std = [0.229, 0.224, 0.225]  # 标准化

    frames = frames*std + mean
    frames = np.clip(frames, 0, 1)

    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(frames[i])
        plt.axis('off')
        plt.title('true:'+class_name[true_labels[i].item()]+'\n'+
                  'pred:'+class_name[pred_labels[i].item()]+'\n'+
                  'scores:'+str(round(pred_scores[i].item(), 3))
                  )

    plt.tight_layout
    plt.show()

model = VIT(num_classes=5)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    for step, (images, labels) in enumerate(dataloader):
        images1, labels = images.to(device), labels.to(device)
        logits = model(images1)
        pred_cls = logits.argmax(dim=1)
        predicts = torch.softmax(logits, dim=1)
        pred_score, _ = predicts.max(dim=1)

        img_show(images, labels, pred_cls, pred_score)



















