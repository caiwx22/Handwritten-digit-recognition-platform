import torchvision


def normal_transform():
    normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return normal


def data_augment_transform():
    data_augment = torchvision.transforms.Compose([
        # 随机裁剪
        # torchvision.transforms.RandomResizedCrop(size=28, scale=(0.4, 1.0)),  # 保持原始大小，缩放比例较小以保留大部分内容
        # 水平翻转
        # torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 较低概率的水平翻转，因为水平翻转对部分数字（如2和5）可能无效
        # 随机旋转
        torchvision.transforms.RandomRotation(degrees=10),  # 旋转角度较小，避免数字混淆
        torchvision.transforms.ToTensor(),
    ])
    return data_augment
