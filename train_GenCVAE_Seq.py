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
from loss import *

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

def train(epoch, beta=1e-2, args=None):
    # train model
    model.train()
    train_loss = 0
    train_CE_loss = 0
    train_KLD_loss = 0

    for batch_idx, (seg, sequence, age, gender, height, weight, sbp, segID) in enumerate(train_dataloader):


        seg = seg.to(device, dtype=torch.float32)
        age, gender, height, weight, sbp= age.to(device), gender.to(device), height.to(device), weight.to(device), sbp.to(device)

        seg = seg.to(device, dtype=torch.float32)


        optimizer.zero_grad()
        recon_batch, disp_seq, mu, logvar = model(seg, age, gender, height, weight, sbp, args)

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
        for batch_idx, (seg, sequence, age, gender, height, weight, sbp, segID) in enumerate(test_dataloader):
            seg= seg.to(device, dtype=torch.float32)
            age, sequence, gender, height, weight, sbp= age.to(device), sequence.to(device), gender.to(device), height.to(device), weight.to(device), sbp.to(device)

            recon_batch, disp_seq, mu, logvar = model(seg, age, gender, height, weight, sbp, args)
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
    parser.add_argument('-c','--condition', type = int, default=5)
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

    model_type = f'cvae-{args.model}-seq-{args.mapping}{args.mapping_number}'


    print(model_type)
    model = networks.GenCVAE_Seq(z_dim=z_dim, img_size=128, depth=64, label_num=label_num, condition=condition,args=args)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model, best_loss = [], 1e10
    # load dataset
    save_path = './Results_CHeart'
    dest_dir = './seg'
    txt_path = './label/train_sequence.txt'
    train_dataset = HamSegData_Condition_4Dvolume(dest_dir,txt_path, debug=False)

    txt_path = './label/val_sequence.txt'
    test_dataset =  HamSegData_Condition_4Dvolume(dest_dir, txt_path, debug=False)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

    # loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_name = '{:s}_zdim_{:d}_epoch_{:d}_beta_{:.2E}_batch_{:d}_lr_{:.2E}_loss_{:s}_ConSeq{:s}.pt'.format(
            model_type, z_dim, N, beta, batch_size, lr, loss_type, args.lstmpos)
    print(model_name)
    folder = 'cvae-seq'
    logdir = os.path.join(f'{save_path}/log/{folder}/', model_name[0:-3])
    writer = SummaryWriter(logdir)
    writer.add_hparams({'type': mzhodel_type, 'z_dim': z_dim, 'epochs': N, 'beta': beta}, {})

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

    writer.close()
