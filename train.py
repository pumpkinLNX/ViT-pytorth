import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ViT import VIT
import numpy as np
# from numpy import load
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  ##dispaly in Chinese

batch_size = 16
epochs = 10
best_loss = 1.0

file_path = 'D:/lnx/code/dataset/flowers recognition/'
weight_path = 'D:/lnx/code/VIT-pretrain model/base_p16_224_backbone.pth'
save_path = 'D:/lnx/code/ViT save weights/'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

data_transformer = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

datasets = {
    'train': datasets.ImageFolder(file_path+'train', transform=data_transformer['train']),
    'val': datasets.ImageFolder(file_path+'val', transform= data_transformer['val'])
}

dataloader = {
    'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True)
}

train_num = len(datasets['train'])
val_num = len(datasets['val'])
print('train_num:', train_num, 'val_num:', val_num)

class_names = dict((v, k) for k, v in datasets['train'].class_to_idx.items())
print('class_names:', class_names)

# train_imgs, train_labels = next(iter(dataloader['train']))
# print('img:', train_imgs.shape, 'labels:', train_labels.shape)
#
# frames = train_imgs[:12]
# frame_labels = train_labels[:12]
# frames = frames.numpy()
# frames = np.transpose(frames, [0, 2, 3, 1])
#
# mean = [0.485, 0.456, 0.406]  # 均值
# std = [0.229, 0.224, 0.225]   # 标准化
# frames = frames * std +mean
# frames = np.clip(frames, 0, 1)
#
# plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.imshow(frames[i])
#     plt.axis('off')
#     plt.title(class_names[frame_labels[i].item()])
# plt.tight_layout
# plt.show()

model = VIT(num_classes=5)
pre_weights = torch.load(weight_path, map_location=device)
print(pre_weights.keys())
##############################
del_keys = ['mlp_head.0.weight', 'mlp_head.0.bias']
for k in del_keys:
    del pre_weights[k]

missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
print('missing_keys:', len(missing_keys), 'unexpected_keys:', len(unexpected_keys))

for params in model.parameters():
    params.requires_grad = True

model.to(device)
loss_function = nn.CrossEntropyLoss()
params_optim = []

for p in model.parameters():
    if p.requires_grad == True:
        params_optim.append(p)

print('training parameters:', len(params_optim))

optimizer = optim.SGD(params_optim, lr=0.001, momentum=0.9, weight_decay=3e-4)

for epoch in range(epochs):
    print('='*30)
    model.train()
    total_loss = 0.0

    for step, (images, labels) in enumerate(dataloader['train']):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)

        loss = loss_function(logits, labels)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if step %100 ==0:
            print(f'step:{step}, loss:{loss}')

    train_loss = total_loss / len(dataloader['train'])

    model.eval()
    total_val_loss = 0
    total_val_corret = 0
    with torch.no_grad():
        for (images, labels) in dataloader['val']:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            total_val_loss += loss

            pred = logits.argmax(dim=1)
            val_correct = torch.eq(pred, labels).float().sum()
            total_val_corret += val_correct

        val_loss = total_val_loss/len(dataloader['val'])
        val_acc = total_val_corret/val_num

        print('-'*30)
        print(f'epoch:{epoch}')
        print(f'train_loss:{train_loss}, val_loss:{val_loss}, val_acc:{val_acc}')

        if train_loss < best_loss:
            save_name = save_path + f'epoch{epoch}_valacc{round(val_acc.item())*100}%_' + 'VIT.pth'
            torch.save(model.state_dict(), save_name)
            best_loss = val_loss
            print(f'Weights have been saved. best_loss has been changed to {val_loss}')









































