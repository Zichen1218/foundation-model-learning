import argparse
from src.plot import plot_rf_depth,plot_samples
from src.data import train_loader,val_loader
from src.config import DEVICE,TRAIN_CONFIG
from src.model import ShallowCNN,DeepCNN,ResNetStyle,DeepCNN_CustomKernel,ResNetStyle_NoSkip
from src.train import train_model


parser = argparse.ArgumentParser(description='CIFAR-10 CNN experiments')
parser.add_argument('mode')
parser.add_argument('name')
parser.add_argument('--ks', type=int)

args = parser.parse_args()
if args.mode == "train":
    if args.name == 'shallow':
        model = ShallowCNN(dropout=0.0).to(DEVICE)
    elif args.name == "deep":
        model = DeepCNN(dropout=0.3).to(DEVICE)
    elif args.name == "resnet":
        model = ResNetStyle(dropout=0.2).to(DEVICE)
    elif args.name == 'resnet_no_skip':
        model = ResNetStyle_NoSkip(dropout=0.2).to(DEVICE)
    elif args.name == 'kernel':
        model = DeepCNN_CustomKernel(kernel_size=args.ks, dropout=0.3).to(DEVICE)
    else:
        print('wrong model name')
    result = train_model(model, train_loader, val_loader, TRAIN_CONFIG, DEVICE)

elif args.mode == "plot":
    if args.name == "rf":
        plot_rf_depth()
    elif args.name == "samples":
        plot_samples()
    else:
        print("wrong plot type")

else:
    print("wrong mode")


