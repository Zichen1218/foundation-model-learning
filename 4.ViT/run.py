import sys
from src.plot import visualize_patches,plot_vit_results,visualize_attention,compare_gap_cls,compare_noPos,visualize_posembd,compare_vit_cnn
from src.train import train_vit_baseline,train_vit_gap,train_vit_noPos,train_cnn

functions = {
    'visualize_patches':visualize_patches,
    'train_vit_baseline':train_vit_baseline,
    'plot_vit_results':plot_vit_results,
    'visualize_attention':visualize_attention,
    'train_vit_gap':train_vit_gap,
    'compare_gap_cls':compare_gap_cls,
    'train_vit_noPos':train_vit_noPos,
    'compare_noPos':compare_noPos,
    'visualize_posembd':visualize_posembd,
    'train_cnn':train_cnn,
    'compare_vit_cnn':compare_vit_cnn,
}

if len(sys.argv) == 1:
    for function in functions:
        print(f"python run.py {function}")
elif sys.argv[1] in functions:
    functions[sys.argv[1]]()
else:
    print('wrong function')