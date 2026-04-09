import sys
from src.plot import plot_rf_depth,plot_samples,compare_architecture,compare_kernel_size,compare_skip,visualize_feature_maps,plot_confusion_matrix
from src.train import train_ShallowCNN,train_DeepCNN,train_ResNet,train_Customkernel,train_ResNet_NoSkip

TRAIN_MODELS = {
    'ShallowCNN':train_ShallowCNN,
    'DeepCNN':train_DeepCNN,
    "ResNet":train_ResNet,
    'Customkernel':train_Customkernel,
    'ResNet_NoSkip':train_ResNet_NoSkip,
}

PLOT_TYPES = {
    'plot_rf_depth':plot_rf_depth,
    'plot_samples':plot_samples,
    'compare_architecture':compare_architecture,
    'compare_kernel_size':compare_kernel_size,
    'compare_skip':compare_skip,
    'visualize_feature_maps':visualize_feature_maps,
    'plot_confusion_matrix':plot_confusion_matrix,
}



def print_help():
    print("ready")
    print()
    print("Available commands:")
    print()

    print("train:")
    for name in TRAIN_MODELS:
        print(f"  python run.py train {name}")
    print()

    print("plot:")
    for name in PLOT_TYPES:
        print(f"  python run.py plot {name}")
    print()

def main():
    argv = sys.argv

    if len(argv) == 1:
        print_help()
        return

    command = argv[1]

    if command == "train":
        if len(argv) < 3:
            print("Please specify a model.")
            print("Available train models:", ", ".join(TRAIN_MODELS.keys()))
            return

        model_name = argv[2]
        if model_name not in TRAIN_MODELS:
            print(f"Unknown model: {model_name}")
            print("Available train models:", ", ".join(TRAIN_MODELS.keys()))
            return

        TRAIN_MODELS[model_name]()

    elif command == "plot":
        if len(argv) < 3:
            print("Please specify a plot type.")
            print("Available plot types:", ", ".join(PLOT_TYPES.keys()))
            return

        plot_name = argv[2]
        if plot_name not in PLOT_TYPES:
            print(f"Unknown plot type: {plot_name}")
            print("Available plot types:", ", ".join(PLOT_TYPES.keys()))
            return

        PLOT_TYPES[plot_name]()

    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__  == "__main__":
    main()


