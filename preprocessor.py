import numpy as np
import json
import torch
from torch.utils.data.dataset import Dataset
import glob
import torchvision.transforms as transforms
from PIL import Image

def parse_json(data):#json文件的读取函数

    arr = np.array([
        data['top'], data['height'], data['left'], data['width'], data['label']
    ])
    arr = arr.astype(int)

    return arr

class SVHNDataset(Dataset):#img数据读取类
    def __init__(self, img_path, img_label, transforms=None):
        self.img_path = img_path
        self.img_label = img_label
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = None

    def __getitem__(self, index):
        img = image.open(self.img_path[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # 设置最长的字符长度为6个
        lbl = np.array(self.img_label[index], dtype=np.int)
        # 使用10填充剩余的位置
        lbl = list(lbl) + (6 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)

def train_transform():
    """
    针对训练集数据的图像处理
    @return:
    """
    return transforms.Compose([
        # 改变图像大小
        transforms.Resize((64, 128)),
        # 调整亮度、对比度、饱和度和色相。brightness是亮度调节因子，contrast对比度参数，saturation饱和度参数，hue是色相因子。
        transforms.ColorJitter(0.3, 0.3, 0.2),
        # 随机旋转图片，degrees表示旋转角度，resample表示重采样方法，expand表示是否扩大图片，以保持原图信息。
        transforms.RandomRotation(5),
        # 将图像转换成张量，同时会进行归一化的一个操作，将张量的值从0-255转到0-1
        transforms.ToTensor(),
        # 数据进行标准化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def Preprocessor(png_dir,json_dir):
    """获取训练集相关数据"""
    # 获取训练集目录下的所有图片
    train_path = glob.glob(png_dir)#训练集png地址
    train_path.sort()
    # 获取训练集图片对应的位置标签和label数据
    train_json = json.load(open(json_dir))#训练集json地址
    train_label = [train_json[x]['label'] for x in train_json]
    # 通过Pytorch加载并处理数据

    train_loader = torch.utils.data.DataLoader(SVHNDataset(train_path, train_label, transforms=train_transform()),
    batch_size=10, shuffle=False, num_workers=0,)
    '''
    batchsize：每批样本个数  shuffle：是否打乱顺序（True/False）   num_workers：读取的线程个数
    '''
    return train_loader

if __name__ == '__main__':
    png_dir = 'smallsize_train/*.png'
    json_dir = 'smallsize_train.json'
    train_loader=Preprocessor(png_dir, json_dir)
    print(train_loader)
