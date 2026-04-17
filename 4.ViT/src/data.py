import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from src.config import config

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=config.batch_size,
                         shuffle=True, num_workers=2, pin_memory=True)
testloader  = DataLoader(testset, batch_size=config.batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')