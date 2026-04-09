from src.train import train_model
from src.config import DEVICE,base_config
from src.data import X_train,y_train
from src.ablation import depth_ablation,width_ablation,dropout_ablation,lr_ablation,count_params
import sys

def run_ablation():
    best_depth = depth_ablation()
    best_width = width_ablation()
    best_droout = dropout_ablation()
    best_lr = lr_ablation()
    return {'best_depth':best_depth,'best_width':best_width,'best_dropout':best_droout,'best_lr':best_lr}

config = base_config
mode = sys.argv[1] if len(sys.argv) > 1 else 'default'
if mode == 'ablation':
    best = run_ablation()
    config['hidden_dims'] = eval(best['best_depth'])
    config['dropout'] = float(best['best_dropout'])
    config['lr'] = float(best['best_lr'])

USE_WANDB = False
best_result = train_model(config, X_train, y_train, DEVICE, use_wandb=USE_WANDB)
print(f'Best val RMSLE: {best_result["best_val_rmsle"]:.4f}')
print(f'Trained for {len(best_result["train_losses"])} epochs')
print(f'Model parameters: {count_params(best_result["model"]):,}')



