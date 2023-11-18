import torch
from torch import optim
from torch.utils.data import DataLoader
from utils import *
from dataset import *
import networks
from image_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')
# from matplotlib.fontmanager import FontProperties
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
from calculate_clinical_features import *

faulthandler.enable()

def VAECELoss(recon_x, x, mu, logvar, disp_seq, beta=1e-2, args = None):
    CE_all = 0
    if 'sw' in args.loss:
        seqweight = torch.tensor(
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(
            device)
    else:
        seqweight = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(
            device)
    if 'cw' in args.loss:
        classweight = torch.tensor([1.0, 2.0, 2.0, 2.0]).to(device)
    else:
        classweight = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)

    for time in range(0, 20):
        x_time = torch.argmax(x[:, time], axis=1)
        CE_t = F.cross_entropy(recon_x[:, time], x_time, reduction='mean', weight=classweight)
        if 'l2' in args.loss:
            CE_all = CE_t * seqweight[time] + CE_all + args.l2u * l2reg_loss(disp_seq[:,time,...])
        else:
            CE_all = CE_t * seqweight[time] + CE_all

    CE = CE_all / torch.sum(seqweight)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return CE + beta * KLD, CE, KLD

def sample_latent_motion(epoch, args, n=2):

    with torch.no_grad():
        phenotypes = ['LVV', 'LVM', 'RVV']

        sample = torch.randn(n, z_dim).to(device)
        age = torch.randint(0, 7, (n,)).to(device)
        gender = torch.randint(0, 2, (n,)).to(device)
        height = torch.randint(160, 180, (n,)).to(device)
        weight = torch.randint(50, 80, (n,)).to(device)
        sbp = torch.randint(100, 130, (n,)).to(device)
        recon_batch, disp_seq = model.decode(sample, age, gender, height, weight, sbp, args)
        recon_batch = recon_batch.cpu()
        disp_seq = disp_seq.cpu()
        #plt.style.use('dark_background')
       # min(seg.size(0), 8)
        if args.disp:
            imagetype = 4
        else:
            imagetype = 3
        scale = 5
        m = n * imagetype + 1  # gt, pred, disp, errormap
        plt.style.use('default')
        fig, axs = plt.subplots(nrows=n * m * 2, ncols=10, frameon=True,
                                gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        fig.set_size_inches([10 * scale, n * m * scale])
        [ax.set_axis_off() for ax in axs.ravel()]
        gender = gender.cpu().detach().numpy()
        age = age.cpu().detach().numpy()

        for k in range(0,n):
            for time in range(0, 10):
                recon_vol = onehot2label(recon_batch[k, time].cpu().detach().numpy())  # debug
                recon_view1 = recon_vol[32, :, :]
                recon_view2 = recon_vol[:, 64, :]
                recon_view3 = recon_vol[:, :, 64]

                axs[k * m, time].imshow(recon_view1, clim=(0, 4))
                axs[k * m + 1, time].imshow(recon_view2, clim=(0, 4))
                axs[k * m + 2, time].imshow(recon_view3, clim=(0, 4))
                if args.disp:
                    disp = disp_seq[k, time, 0:2, 32, :, :].cpu().detach().numpy()
                    # warped grid: ground truth
                    ax = axs[k * m + 3, time]
                    plot_warped_grid(ax, disp, interval=8, title="disp_xy", fontsize=20)


            for time in range(10, 20):
                time_axis = time - 10
                recon_vol = onehot2label(recon_batch[k, time].cpu().detach().numpy())  # debug
                recon_view1 = recon_vol[32, :, :]
                recon_view2 = recon_vol[:, 64, :]
                recon_view3 = recon_vol[:, :, 64]

                axs[k * m + imagetype, time_axis].imshow(recon_view1, clim=(0, 4))
                axs[k * m + imagetype + 1, time_axis].imshow(recon_view2, clim=(0, 4))
                axs[k * m + imagetype + 2, time_axis].imshow(recon_view3, clim=(0, 4))
                # warped grid: ground truth
                if args.disp:
                    disp = disp_seq[k, time, 0:2, 32, :, :].cpu().detach().numpy()
                    ax = axs[k * m + imagetype + 3, time_axis]
                    plot_warped_grid(ax, disp, interval=8, title="disp_xy", fontsize=20)


            # LVV, LVM, RVV cureve
            time = range(0, 20)
            Feature_curve = calculate_FeatureCurve(onehot2label(recon_batch[k].cpu().detach().numpy(),axis=1))
            for idx, phenotype in enumerate(phenotypes):
                axs[k * m + imagetype * 2, idx].plot(time, np.array(Feature_curve[phenotype]))
                axs[k * m + imagetype * 2, 0].set_title(f'{phenotype}-pred', fontsize=10)

            if gender[k] == 0:
                SEX = 'female'

            if gender[k] == 1:
                SEX = 'male'
            axs[k * m, 0].set_title(f'{SEX}, {age[k]}', size='small',family='scan-serif')

        writer.add_figure('Latent sampling', fig, epoch)


def sample_latent_age(epoch, args=None, n=8):

    with torch.no_grad():
        # plt.style.use('dark_background')
        plt.style.use('default')
        fig, axs = plt.subplots(nrows=6, ncols=n, frameon=False,
                                gridspec_kw={'wspace': 0.5, 'hspace': 0.05})
        fig.set_size_inches([n, 6])
        [ax.set_axis_off() for ax in axs.ravel()]
        sample = torch.randn(1, z_dim).to(device)
        gender = torch.randint(0, 2, (1,)).to(device)
        height = torch.randint(160, 180, (1,)).to(device)
        weight = torch.randint(50, 80, (1,)).to(device)
        sbp = torch.randint(100, 130, (1,)).to(device)

        for k in range(0, n):
            age = torch.from_numpy(np.array([k])).to(device)
            #age = torch.randint(0, 7, (n,)).to(device)
            sample_result, disp_seq = model.decode(sample, age, gender, height, weight, sbp, args)
            gendercopy = gender
            gendercopy = gendercopy.cpu().detach().numpy()
            agecopy = age
            agecopy = agecopy.cpu().detach().numpy()
            vol_onehot = onehot2label(sample_result[0, 0].cpu().detach().numpy())

            if k ==0:
                error = np.zeros((64,128,128))
                vol_onehot_last = vol_onehot
            else:

                error = vol_onehot-vol_onehot_last

            view1 = vol_onehot[32, :, :]
            view2 = vol_onehot[:, 64, :]
            view3 = vol_onehot[:, :, 64]
            error1 = error[32, :, :]
            error2 = error[:, 64, :]
            error3 = error[:, :, 64]

            axs[0, k].imshow(view1, clim=(0, 4))
            axs[2, k].imshow(view2, clim=(0, 4))
            axs[4, k].imshow(view3, clim=(0, 4))

            axs[1, k].imshow(error1, clim=(-8, 8),cmap='seismic')
            axs[3, k].imshow(error2, clim=(-8, 8),cmap='seismic')
            axs[5, k].imshow(error3, clim=(-8, 8),cmap='seismic')

            if gendercopy[0]== 0:
                SEX = 'female'

            if gendercopy[0] == 1:
                SEX = 'male'

            axs[0, k].set_title(f'{agecopy[0]}, {SEX}', size='small',
                                family='scan-serif')

        writer.add_figure('Latent sampling age', fig, epoch)


def shot_latent(epoch, mu, logvar):
    plt.style.use('dark_background')
    fig = plt.figure()
    X_embedded = mu[:, 0:2]
    sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, c=np.exp(logvar).mean(axis=1))
    plt.colorbar(sc)
    plt.axvline(0, color='r')
    plt.axhline(0, color='r')
    plt.title('latent space - raw', fontsize=16)
    writer.add_figure('Latent view', fig, epoch)


def train(epoch, beta=1e-2, args=None):
    # train model
    model.train()
    train_loss = 0
    train_CE_loss = 0
    train_KLD_loss = 0

    for batch_idx, (seg, age, age_group_target, gender, hypertension, segID) in enumerate(train_dataloader):


        seg = seg.to(device, dtype=torch.float32)
        age, gender, hypertension = age.to(device), gender.to(device), hypertension.to(device)

        seg = seg.to(device, dtype=torch.float32)


        optimizer.zero_grad()
        recon_batch, disp_seq, mu, logvar = model(seg, age, gender, hypertension, args)

        loss, CE, KLD = VAECELoss(recon_batch, seg, mu, logvar, disp_seq, beta=beta, args=args)

        loss.backward()
        optimizer.step()

        # loss update and log
        train_loss += loss.item()
        train_CE_loss += CE.item()
        train_KLD_loss += KLD.item()
        if batch_idx % int(len(train_dataloader.dataset)/2) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(seg), len(train_dataloader.dataset),
                       100. * batch_idx * len(seg) / len(train_dataloader),
                       loss.item()))

    train_loss /= batch_idx+1
    train_CE_loss /=batch_idx+1
    train_KLD_loss /= batch_idx + 1
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    return train_loss, train_CE_loss, train_KLD_loss


def test(epoch, model=None, beta=1e-2, args=None):
    feature_names = ['LVV', 'LVM', 'RVV']
    model.eval()
    latent_mu, latent_logvar = [], []
    test_loss = 0
    test_CE_loss = 0
    test_KLD_loss = 0
    with torch.no_grad():
        for batch_idx, (seg, age, age_group_target, gender, hypertension, segID) in enumerate(test_dataloader):
            seg= seg.to(device, dtype=torch.float32)
            age, gender, hypertension = age.to(device), gender.to(device), hypertension.to(device)

            recon_batch, disp_seq, mu, logvar = model(seg, age, gender, hypertension, args)
            # record coordinates
            mu_temp = mu.cpu().detach().numpy()
            latent_mu.append(mu_temp)
            logvar_temp = logvar.cpu().detach().numpy()
            latent_logvar.append(logvar_temp)

            t_loss, CE, KLD = VAECELoss(recon_batch, seg, mu, logvar, disp_seq, beta=beta, args=args)

            # loss update and log
            test_loss += t_loss.item()
            test_CE_loss += CE.item()
            test_KLD_loss += KLD.item()
            # if batch_idx == 0:
            #     n = 2  # min(seg.size(0), 8)
            #     if args.disp:
            #         imagetype = 4
            #     else:
            #         imagetype = 3
            #     scale = 5
            #     m = n * imagetype + 1 #gt, pred, disp, errormap
            #     plt.style.use('default')
            #     fig, axs = plt.subplots(nrows=n * m * 2, ncols=10, frameon=True,
            #                             gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
            #     fig.set_size_inches([10 * scale, n * m * scale])
            #     [ax.set_axis_off() for ax in axs.ravel()]
            #
            #     for k in range(0, n):
            #         for time in range(0, 10):
            #             seg_vol = onehot2label(seg[k, time].cpu().detach().numpy())
            #             recon_vol = onehot2label(recon_batch[k, time].cpu().detach().numpy())  # debug
            #             dsc = np.mean(np_mean_dice(recon_vol, seg_vol))
            #
            #             view1 = seg_vol[32, :, :]
            #             recon_view1 = recon_vol[32, :, :]
            #             error = view1 - recon_view1
            #
            #
            #             axs[k * m, time].imshow(view1, clim=(0, 4))
            #             axs[k * m + 1, time].imshow(recon_view1, clim=(0, 4))
            #             axs[k * m + 2, time].imshow(error, clim=(-8, 8), cmap='seismic')
            #             axs[k * m, time].set_title('{:0.2f}'.format(dsc))
            #             # warped grid: ground truth
            #             if args.disp:
            #                 disp = disp_seq[k, time, 0:2, 32, :, :].cpu().detach().numpy()
            #                 ax = axs[k * m + 3, time]
            #                 plot_warped_grid(ax, disp, interval=8, title="disp_xy", fontsize=20)
            #
            #         for time in range(10, 20):
            #             seg_vol = onehot2label(seg[k, time].cpu().detach().numpy())
            #             recon_vol = onehot2label(recon_batch[k, time].cpu().detach().numpy())  # debug
            #             dsc = np.mean(np_mean_dice(recon_vol, seg_vol))
            #             view1 = seg_vol[32, :, :]
            #             recon_view1 = recon_vol[32, :, :]
            #             error = view1 - recon_view1
            #
            #             time_axis = time - 10
            #
            #             axs[k * m + imagetype, time_axis].imshow(view1, clim=(0, 4))
            #             axs[k * m + imagetype + 1, time_axis].imshow(recon_view1, clim=(0, 4))
            #             axs[k * m + imagetype + 2, time_axis].imshow(error, clim=(-8, 8), cmap='seismic')
            #             axs[k * m + imagetype, time_axis].set_title('{:0.2f}'.format(dsc))
            #             # warped grid: ground truth
            #             if args.disp:
            #                 disp = disp_seq[k, time, 0:2, 32, :, :].cpu().detach().numpy()
            #                 ax = axs[k * m + imagetype + 3, time_axis]
            #                 plot_warped_grid(ax, disp, interval=8, title="disp_xy", fontsize=20)
            #
            #         # LVV, LVM, RVV cureve
            #         time = range(0, 20)
            #         Feature_curve_gt =calculate_FeatureCurve(seg[k].cpu().detach().numpy())
            #         Feature_curve_pre = calculate_FeatureCurve(recon_batch[k].cpu().detach().numpy())
            #
            #         for idx, fn in enumerate(feature_names):
            #             axs[k * m + imagetype * 2, idx * 2].plot(time, np.array(Feature_curve_gt[f'{fn}']))
            #             axs[k * m + imagetype * 2, idx * 2 + 1].plot(time, np.array(Feature_curve_pre[f'{fn}']))
            #             axs[k * m + imagetype * 2, idx*2].set_title(f'{fn}-gt', fontsize=10)
            #             axs[k * m + imagetype * 2, idx*2+1].set_title(f'{fn}-pred', fontsize=10)
            #
            #         writer.add_figure('Reconstruction', fig, epoch)
    test_loss /= batch_idx+1
    test_CE_loss /= batch_idx + 1
    test_KLD_loss /= batch_idx + 1
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, np.concatenate(latent_mu), np.concatenate(latent_logvar), test_CE_loss, test_KLD_loss


if __name__ == '__main__':

    if 0: # debug
        z_dim = 8
        lr = 1e-4
        batch_size = 4
        beta = 1e-2
        N = 100
        gpu_id = 1


    parser = argparse.ArgumentParser(description='C-VAE-4D')
    parser.add_argument('--z_dim', default=32, required=True, type=int)
    parser.add_argument('-b','--batch_size', default=4, required=True)
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('-m', '--model', type=str, default='lstmcell')
    parser.add_argument('--beta', default=1e-2)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--gpu', default=0)

    parser.add_argument('--label_num', default=4)
    parser.add_argument('-c','--condition', type = int, default=3)
    parser.add_argument('-p', '--pre_train', type = str, default=None)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--l2u', type=float, default=0.01)
    parser.add_argument('--disp', default=False)
    parser.add_argument('--arch', type=str, default='none', choices=['none', 'condisp', 'disp20'])
    parser.add_argument('--mapping', type=str, default='none', choices=['none', 'En', 'De', 'EnDe'])
    parser.add_argument('--mapping_number', type=int,
                        default=0)  # action='store_true', help='ade mapping latent age code')

    parser.add_argument('--visuallatent', default=False)  ### TODO: check the output to make sure there is no disp
    parser.add_argument('--lstmpos', type=str, default='after', choices=['after', 'before'],
                        help='the position to incorporate the conditions into sequential latent space')

    #pre_train model list:cvae_EDES,cvae
    args = parser.parse_args()

    # set visible GPU env
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # hyperparameter
    z_dim = int(args.z_dim)
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    beta = float(args.beta)
    loss_type = args.loss
    N = int(args.epochs)
    condition = int(args.condition)

    label_num = int(args.label_num)

    if args.mapping == 'none':
        model_type = f'cvae-{args.model}-seq'
    else:
        model_type = f'cvae-{args.model}-seq-{args.mapping}{args.mapping_number}'


    print(model_type)
    # Oct4->new, before->Seq_0
    if args.arch == 'condisp':
        model = networks.GenCVAE_Seq_dsp_con(z_dim=z_dim, img_size=128, depth=64, label_num=label_num,
                                             condition=3)
    else:
        model = networks.GenCVAE_Seq_UKBB_edes(z_dim=z_dim, img_size=128, depth=64, label_num=label_num, condition=condition,args=args)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model, best_loss = [], 1e10
    # load dataset
    save_path = '/vol/biomedic3/mq21/code/Results_CHeart'
    dest_dir = '/vol/biodata/data/MeshHeart_small_set/'
    txt_path = f'/vol/biomedic3/mq21/data/cardiac/UKbiobank/label/mesh_40616/mesh_train.csv'
    train_dataset = UKbiobank_40k_EDES_cycle(dest_dir, txt_path)
    txt_path = f'/vol/biomedic3/mq21/data/cardiac/UKbiobank/label/mesh_40616/mesh_test.csv'
    test_dataset = UKbiobank_40k_EDES_cycle(dest_dir, txt_path)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

    # loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if 'after' in args.lstmpos:
        model_name = '{:s}_zdim_{:d}_epoch_{:d}_beta_{:.2E}_batch_{:d}_lr_{:.2E}_loss_{:s}.pt'.format(model_type, z_dim,
                                                                                                      N, beta,
                                                                                                      batch_size, lr,
                                                                                                      loss_type)
    else:
        model_name = '{:s}_zdim_{:d}_epoch_{:d}_beta_{:.2E}_batch_{:d}_lr_{:.2E}_loss_{:s}_ConSeq{:s}.pt'.format(
            model_type, z_dim, N, beta, batch_size, lr, loss_type, args.lstmpos)
    print(model_name)
    folder = 'cvae-seq-2023-UKBB'# save model after 300, every 25
    logdir = os.path.join(f'{save_path}/log/{folder}/', model_name[0:-3])
    writer = SummaryWriter(logdir)
    writer.add_hparams({'type': model_type, 'z_dim': z_dim, 'epochs': N, 'beta': beta}, {})

    # train
    for epoch in tqdm(range(0, N)):

        train_loss, train_CE_loss, train_KLD_loss = train(epoch, beta, args)
        test_loss, test_mu, test_logvar, test_CE_loss, test_KLD_loss = test(epoch, model=model,beta=beta, args=args)
        modelpath = f'{save_path}/models/{folder}/{model_name[0:-3]}/'
        setup_dir(modelpath)
        # update best loss
        if test_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = test_loss

            torch.save(best_model.state_dict(),
                       os.path.join(modelpath, 'best_loss.pt'))
        if (epoch>249) & (epoch % 25 == 0):
            epoch_model = copy.deepcopy(model)
            torch.save(epoch_model.state_dict(),
                       os.path.join(modelpath, f'epoch{epoch}.pt'))
            # setup_dir(f'./models/{folder}/')
            # torch.save(best_model.state_dict(),
            #            os.path.join(f'./models/{folder}/', model_name))

        # segID = '14CD03161'
        # image_name = '{0}/{1}_{2:02d}.npy'.format(dest_dir, segID, 0)
        # seg = np.load(image_name, allow_pickle=True)
        # ED = label2onehot(seg)
        # sample_latent_motion(epoch, args, 2)
        # sample_latent_age(epoch, args, 8)
        # # sample_latent(epoch, args, ED, 1)
        # shot_latent(epoch, test_mu, test_logvar)
        #
        # writer.add_scalars('Loss', {'train': train_loss,
        #                             'test': test_loss}, epoch)
        # writer.add_scalars('CE-Loss', {'train': train_CE_loss,
        #                             'test': test_CE_loss}, epoch)
        # writer.add_scalars('KLD-Loss', {'train': train_KLD_loss,
        #                             'test': test_KLD_loss}, epoch)

    writer.close()
