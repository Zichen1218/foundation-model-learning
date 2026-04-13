import torch
import numpy as np
import sys
from src.function import train_gpt,generate_text,ablation_depth_width,train_gpt_noPos,train_GPT_noCausal
from src.plot import plot_gpt_result,plot_ablation_depth_width,ablation_Pos,ablation_noCausal,plot_attention_maps,compare_temperature

torch.manual_seed(42)
np.random.seed(42)

train_functions = ['train_gpt','generate_text','ablation_depth_width','train_gpt_noPos','train_GPT_noCausal']
plot_functions = ['plot_gpt_result','plot_ablation_depth_width','ablation_Pos','ablation_noCausal',
                  'plot_attention_maps','compare_temperature']
def main():
    if len(sys.argv) == 1:
        print('available commands:\n')
        for func in train_functions:
            print(f'python run.py train '+func)
        for func in plot_functions:
            print(f'python run.py plot '+func)
    elif sys.argv[1] == 'train':
        if sys.argv[2] == 'train_gpt':
            train_gpt()
        elif sys.argv[2] =='generate_text':
            generate_text()
        elif sys.argv[2] =='ablation_depth_width':
            ablation_depth_width()
        elif sys.argv[2] == 'train_gpt_noPos':
            train_gpt_noPos()
        elif sys.argv[2] == 'train_GPT_noCausal':
            train_GPT_noCausal()
        else:
            print('wrong model')

    elif sys.argv[1] == 'plot':
        if sys.argv[2] == 'plot_gpt_result':
            plot_gpt_result()
        elif sys.argv[2] =='plot_ablation_depth_width':
            plot_ablation_depth_width()
        elif sys.argv[2] =='ablation_Pos':
            ablation_Pos()
        elif sys.argv[2] =='ablation_noCausal':
            ablation_noCausal()
        elif sys.argv[2] =='plot_attention_maps':
            plot_attention_maps()
        elif sys.argv[2] =='compare_temperature':
            compare_temperature()
        else:
            print('wrong plot arguments')

    else:
        print('wrong command.')

if __name__ == '__main__':
    main()