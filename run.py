import argparse
import random
from datetime import datetime

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_few_shot_forecasting import Exp_Few_Shot_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from utils.print_args import print_args

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='CMIE')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='CMIE',
                        help='model name, options: [CMIE,CALF]')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument("--epsilon", type=float, default=0, help="noise")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")

    # model define
    parser.add_argument("--gpt_layers", type=int, default=6, help="number of GPT layers")
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument("--merge_layers",type=int, default=1, help='num of dataMerge layers')
    parser.add_argument('--merge_threshold', type=float, default=0.0, help='distance threshold for adaptive channel patching in ChannelMerge (0 = disable; larger -> fewer patches)')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--output_loss', type=str, default='l1', help='loss function')
    parser.add_argument('--lambda_sim', type=float, default=0.1, help='weight for embedding similarity regularization L_sim (default 0.1; set 0 to disable)')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu (0 or 1)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # ablation flags
    parser.add_argument('--use_ffc', type=int, default=1, help='use full-field convolution (1) or linear embedding (0)')
    parser.add_argument('--use_channel_aware_prompt', type=int, default=1,
                        help='use channel-aware prompts (1) or generic prompt (0)')
    parser.add_argument('--use_llm', type=int, default=1, help='use LLM backbone (1) or bypass (0)')
    parser.add_argument('--llm_backbone', type=str, default='gpt2', choices=['gpt2', 'transformer'],
                        help='LLM backbone type (gpt2 or transformer encoder)')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # CALF word embedding
    parser.add_argument('--word_embedding_path', type=str, default='./word_emb.pt',
                        help='path to word embedding .pt file for CALF')

    # self
    parser.add_argument('--tmax', type=int, default=20)
    parser.add_argument('--cos', type=int, default=1)

    # few shot percentage
    parser.add_argument('--percent', type=int, default=100, help='few shot percentage')

    # zero shot
    parser.add_argument('--zero_shot', type=int, default=0, help='1 for training zero shot')
    parser.add_argument('--target_data', type=str, default="ETTm1", help='target data for zero shot')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False



    print('Args in experiment:')
    print_args(args)

    if "ETT" in args.data:
        args.N = 7
    elif "Weather" in args.data:
        args.N = 21
    elif "Traffic" in args.data:
        args.N = 862
    elif "ECL" in args.data:
        args.N = 321
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'few_shot_forecast':
        Exp = Exp_Few_Shot_Forecast
    elif args.task_name == 'zero_shot_forecast':
        Exp = Exp_Zero_Shot_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_topk{}_r{}_lora{},{}_gptlayer{}_dmodel{}_nheads{}_dlayers{}_merge{}_epsilon{}_decay{}_dropout{}_ii{}'.format(
                args.task_name,
                args.model_id,
                args.top_k,
                args.r,
                args.lora_alpha,
                args.lora_dropout,
                args.gpt_layers,
                args.d_model,
                args.n_heads,
                args.d_layers,
                args.merge_layers,
                args.epsilon,
                args.weight_decay,
                args.dropout,
                ii)

            # Append ablation suffix
            ablation_suffix = ""
            if not args.use_ffc:                ablation_suffix += "_noFFC"
            if not args.use_channel_aware_prompt: ablation_suffix += "_noCAP"
            if not args.use_llm:                ablation_suffix += "_noLLM"
            if args.llm_backbone != 'gpt2':      ablation_suffix += "_transformer"
            if args.merge_threshold > 0:            ablation_suffix += f"_thr{args.merge_threshold}"
            setting += ablation_suffix

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_topk{}_r{}_lora{},{}_gptlayer{}_dmodel{}_nheads{}_dlayers{}_merge{}_epsilon{}_decay{}_dropout{}_ii{}'.format(
            args.task_name,
            args.model_id,
            args.top_k,
            args.r,
            args.lora_alpha,
            args.lora_dropout,
            args.gpt_layers,
            args.d_model,
            args.n_heads,
            args.d_layers,
            args.merge_layers,
            args.epsilon,
            args.weight_decay,
            args.dropout,
            ii)

        ablation_suffix = ""
        if not args.use_ffc:                ablation_suffix += "_noFFC"
        if not args.use_channel_aware_prompt: ablation_suffix += "_noCAP"
        if not args.use_llm:                ablation_suffix += "_noLLM"
        if args.llm_backbone != 'gpt2':      ablation_suffix += "_transformer"
        if args.merge_threshold > 0:            ablation_suffix += f"_thr{args.merge_threshold}"
        setting += ablation_suffix

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

# 获取当前时间
now = datetime.now()
# 格式化输出
formatted_time = now.strftime("%Y年%m月%d日 %H:%M")
print(formatted_time)
