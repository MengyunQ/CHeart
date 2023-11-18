import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import *
import networks
from image_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
import os
from torch.utils.tensorboard import SummaryWriter
import copy
import argparse
import numpy as np
import torch.nn.functional as F
import random
from textwrap import wrap
import faulthandler
from visualize import *
from loss import *
from utils import *
from Metric_utils import *
from Evaluation_of_generation import *
from perform_age_distribution import *
faulthandler.enable()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='C-VAE-4D')
    parser.add_argument('--z_dim', default=8, type=int,required=True)
    parser.add_argument('-bm', '--batch_size_inmodel', default=6, required=True)
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('-m', '--model', type=str, default='lstmcell')
    parser.add_argument('--beta', default=1e-2)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--disp', default=False)
    parser.add_argument('--label_num', default=4)
    parser.add_argument('-c', '--condition', type=int, default=5)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--l2u', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default='none', choices=['none','condisp','disp20'])
    parser.add_argument('--metric', type=str, default = 'all', choices=['all', 'recon',
                                                                        'c-fea-test', 'sample-generation',
                                                                        'plot-sample', 'age','tsne','recon_base'])
    parser.add_argument('--modelname', type=str, default='best_loss', required=True)
    parser.add_argument('--gan', default='vae', choices=['seq', 'vae'])
    parser.add_argument('--vaegan', default=False)
    parser.add_argument('--mapping', type=str, default='none', choices=['none', 'En', 'De', 'EnDe'])
    parser.add_argument('--mapping_number', type=int,
                        default=0)  # action='store_true', help='ade mapping latent age code')

    parser.add_argument('--visuallatent', default=False)  ### TODO: check the output to make sure there is no disp
    parser.add_argument('--lstmpos', type=str, default='after', choices=['after', 'before'],
                        help='the position to incorporate the conditions into sequential latent space')

    # pre_train model list:cvae_EDES,cvae
    args = parser.parse_args()
    print(f'calculating{args.metric}')

    # set visible GPU env
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # hyperparameter
    z_dim = int(args.z_dim)
    lr = float(args.lr)
    batch_size_inmodel = int(args.batch_size_inmodel)
    beta = float(args.beta)
    loss_type = args.loss
    N = int(args.epochs)

    label_num = int(args.label_num)
    condition = int(args.condition)
    if args.mapping == 'none':
        model_type = f'cvae-{args.model}-seq'
    else:
        model_type = f'cvae-{args.model}-seq-{args.mapping}{args.mapping_number}'

    print(model_type)
    # Oct4->new, before->Seq_0
    if args.arch == 'condisp':
        model = networks.GenCVAE_Seq_dsp_con(z_dim=z_dim, img_size=128, depth=64, label_num=label_num,
                                             condition=condition)
    else:
        model = networks.GenCVAE_Seq(z_dim=z_dim, img_size=128, depth=64, label_num=label_num, condition=condition,args=args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder = 'cvae-seq-2022-retry'
    plot_number = 5
    save_path = '/vol/biomedic3/mq21/code/Results_CHeart'
    dest_dir = '/vol/biomedic3/mq21/data/cardiac/GenScan/H4D/seg'
    txt_path = '/vol/biomedic3/mq21/data/cardiac/GenScan/label/test_sequence.txt'
    model_path = f'/vol/biomedic3/mq21/code/Results_CHeart/models/{folder}'
    test_dataset = HamSegData_Condition_4Dvolume(dest_dir, txt_path, debug=False)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=2)

    if 'after' in args.lstmpos:
        model_name = '{:s}_zdim_{:d}_epoch_{:d}_beta_{:.2E}_batch_{:d}_lr_{:.2E}_loss_{:s}.pt'.format(model_type, z_dim,
                                                                                                      N, beta,
                                                                                                      batch_size_inmodel, lr,
                                                                                                      loss_type)
    else:
        model_name = '{:s}_zdim_{:d}_epoch_{:d}_beta_{:.2E}_batch_{:d}_lr_{:.2E}_loss_{:s}_ConSeq{:s}.pt'.format(
            model_type, z_dim, N, beta, batch_size_inmodel, lr, loss_type, args.lstmpos)
    print(model_name)

    modelnumber = args.modelname
    model_name_all = f'{model_path}/{model_name[0:-3]}/{modelnumber}.pt'
    print(model_name_all)
    model.load_state_dict(torch.load(model_name_all))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    resultpath = f'{save_path}/results/{folder}/{model_name[0:-3]}/{modelnumber}/'
    setup_dir(resultpath)


    # 1. Reconstruction loss
    if args.metric == 'all' or args.metric == 'recon':
        reconpath = f'{resultpath}recon/'
        setup_dir(reconpath)
        report_data_dict = val(reconpath, plot_number=plot_number,
                                           args=args, model=model, device=device, test_dataloader=test_dataloader)
        df = pd.DataFrame(data=report_data_dict)
        df.to_pickle(reconpath + f'/analysis_results_df.pkl')


    # 3. mutiple samples, same conditions for Clinical Features Analysis
    # mean value of each clinical feature in each condition(agegroup, gender, ...), with other random conditions
    # analysis the linear regression and the range of a specific condition
    Dice_sample, HD_sample, ASSD_sample =[], [], []
    if args.metric == 'sample-generation' or args.metric == 'all':
        print('start sample generation')
        genpath = f'{resultpath}/generation/'
        setup_dir(genpath)

        sample_number = 5

        results_acc0, clinical_val_real0, clinical_val_syn0, condition_results0, condition_results_syn0\
            = same_condition_clinical_feature_multisamples(txt_path = txt_path, args=args, sample_number=sample_number, model=model,
                                                           device=device, dest_dir=dest_dir)
        results_acc1, clinical_val_real1, clinical_val_syn1, condition_results1, condition_results_syn1 \
            = same_condition_clinical_feature_multisamples(txt_path = txt_path, args=args, sample_number=sample_number, model=model,
                                                           device=device, dest_dir=dest_dir)
        results_acc2, clinical_val_real2, clinical_val_syn2, condition_results2, condition_results_syn2 \
            = same_condition_clinical_feature_multisamples(txt_path = txt_path, args=args, sample_number=sample_number, model=model,
                                                           device=device, dest_dir=dest_dir)
        results_acc3, clinical_val_real3, clinical_val_syn3, condition_results3, condition_results_syn3 \
            = same_condition_clinical_feature_multisamples(txt_path = txt_path, args=args, sample_number=sample_number, model=model,
                                                           device=device, dest_dir=dest_dir)
        #concat val of synthetic data
        metrics = ['Dice', 'HD', 'ASSD']
        types = ['min', 'max', 'mean']
        features = ['LVEDV (mL)','LVM (g)', 'RVEDV (mL)', 'LVESV', 'LVESM', 'RVESV', 'LVSV', 'LVCO', 'LVEF (%)', 'RVSV', 'RVCO',
                    'RVEF (%)']
        results_acc = {'ID': []}
        for metric in metrics:
            for type in types:
                results_acc.update({f'{metric}_allcls_{type}': []})
                for cls in range(3):
                    results_acc.update({f'{metric}_cls_{cls}_{type}': []})

        for feature in features:
            for type in types:
                results_acc.update({f'distance{feature}_{type}': []})
        for item in results_acc0['ID']:
            index = results_acc0['ID'].index(item)
            for key in results_acc0.keys():
                if key=='ID':
                    results_acc[key].append(item)
                if 'max' in key:
                    results_acc[key].append(np.max([results_acc0[key][index],results_acc1[key][index],results_acc2[key][index],results_acc3[key][index]]))
                if 'min' in key:
                    results_acc[key].append(np.min([results_acc0[key][index],results_acc1[key][index],results_acc2[key][index],results_acc3[key][index]]))
                if 'mean' in key:
                    results_acc[key].append(np.mean([results_acc0[key][index],results_acc1[key][index],results_acc2[key][index],results_acc3[key][index]]))

        df = pd.DataFrame.from_dict(results_acc)
        acc_txtfile = f'{genpath}/generation-multisample-acc.csv'
        df.to_csv(acc_txtfile)

        clinical_val_syn = {
            key: clinical_val_syn0[key] + clinical_val_syn1.get(key, '') + clinical_val_syn2.get(key, '') + clinical_val_syn3.get(key, '')
            for key in clinical_val_syn0.keys()}
        recon_df = pd.DataFrame.from_dict(clinical_val_syn)
        csvfile2 = f'{genpath}/clinical_val_syn.csv'
        recon_df.to_csv(csvfile2, header=True, index=True)

        condition_results_syn = {
            key: condition_results_syn0[key] + condition_results_syn1.get(key, '')
                 + condition_results_syn2.get(key, '') + condition_results_syn3.get(key, '')
            for key in condition_results_syn0.keys()}
        condition_syn = pd.DataFrame.from_dict(condition_results_syn)
        csvfile4 = f'{genpath}/condition_syn.csv'
        condition_syn.to_csv(csvfile4, header=True, index=True)

        # real data
        condition_real = pd.DataFrame.from_dict(condition_results0)
        csvfile3 = f'{genpath}/condition_real.csv'
        condition_real.to_csv(csvfile3, header=True, index=True)
        val_df = pd.DataFrame.from_dict(clinical_val_real0)
        csvfile1 = f'{genpath}/clinical_val_real.csv'
        val_df.to_csv(csvfile1, header=True, index=True)
        #draw the distribution plot
        plot_clinical_feature_all(data_path=genpath, samplenum=sample_number*4)
        print('clinical-analysis-end')

    # 4. plot comparison images between testing dataset and synthetic dataset
    if args.metric == 'all' or args.metric == 'plot-sample':
        print('plot-sample')
        plotpath = f'{resultpath}plot/'
        setup_dir(plotpath)
        fh = open(txt_path, 'r')
        plot_number = 5
        plot_multisamples(fh, plot_number=plot_number, resultpath=plotpath,args=args,
                          model=model, device=device, dest_dir=dest_dir)


    # 5. how a cardiac changes through one condition with other conditions fixed (such as age, gender)
    if args.metric == 'all' or args.metric == 'age':
        print('generation through age changing')
        plotpath = f'{resultpath}plot/'
        setup_dir(plotpath)
        fh = open(txt_path, 'r')
        generation_through_age(resultpath=plotpath, args=args, model=model, device=device)

    if args.metric == 'all' or args.metric == 'tsne':
        txt_path = '/vol/biomedic3/mq21/data/cardiac/GenScan/label/train_sequence.txt'
        train_dataset = HamSegData_Condition_4Dvolume(dest_dir, txt_path, debug=False)
        train_dataloader = DataLoader(train_dataset, 1, shuffle=False, num_workers=2)
        print('tsne visualization')
        resultpath_tsne = '/vol/biomedic3/mq21/code/Results_CHeart/visual_jupyter/latentspace'
        setup_dir(resultpath_tsne)
        fh = open(txt_path, 'r')
        report_data_dict = latent_space_visual(resultpath_tsne, args=args, model=model, device=device, test_dataloader=train_dataloader)

    if args.metric == 'recon_base':
        reconpath = f'{resultpath}recon/'
        setup_dir(reconpath)
        report_data_dict = val_base_apex(reconpath, plot_number=plot_number,
                                           args=args, model=model, device=device, test_dataloader=test_dataloader)
        df = pd.DataFrame(data=report_data_dict)
        df.to_pickle(reconpath + f'/analysis_results_df_base_apex.pkl')