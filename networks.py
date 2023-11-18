import torch
from torch import nn
from utils import idx2onehot
from image_utils import onehot2label,label2onehot
from transformer import warp
import torch.nn.functional as F

# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, D, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.D, self.H, self.W = C, D, H, W

    def forward(self, input):
        return input.view(input.size(0), self.C, self.D, self.H, self.W)

#4D volume generation
# the first of lstm is z, Oct4
class GenCVAE_Seq(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    ### This is a clean version Oct for test sequential condition
    def __init__(self, img_size=128, z_dim=8, nf=4, depth=64,label_num=4, condition = 5,args=None):
        super(GenCVAE_Seq, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n, channel:4labels + 6 conditions
        if 'En' in args.mapping:
            self.conv1 = nn.Conv3d(label_num + args.mapping_number, nf, kernel_size=4, stride=2,
                                   padding=1)  # layer size increasing
        else:
            self.conv1 = nn.Conv3d(label_num+condition, nf, kernel_size=4, stride=2, padding=1)  # layer size increasing
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf * 8 * depth / 16 * img_size / 16 * img_size / 16)

        self.linear_means = nn.Linear(h_dim, z_dim)
        self.linear_log_var = nn.Linear(h_dim, z_dim)
        if 'De' in args.mapping:
            self.fc2 = nn.Linear(z_dim + args.mapping_number, h_dim)
        else:
            self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.fc_disp = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)


        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16), H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
        )

        if 'after' in args.lstmpos:
            self.input_size = z_dim
        elif 'De' in args.mapping:
            self.input_size = z_dim + args.mapping_number
        else:
            self.input_size = z_dim + args.condition
        # self.input_size = z_dim
        self.length_sequence = 20
        self.hidden_size = self.input_size
        self.num_layers = 2
        # self.lstm = nn.LSTM(
        #     input_size = self.input_size,
        #     hidden_size = self.hidden_size,
        #     num_layers = self.num_layers,
        #     batch_first = True,
        #     # dropout=0.2,
        # )
        # self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
        #                       batch_first=True)
        # # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        # self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        if args.mapping!='none':
            self.mapping = MLP(condition, args.mapping_number, 64, 8, weight_norm=True)
        # self.mapping = MLP(condition, args.mapping_number, 64, 8, weight_norm=True) ###TODO: should be deleted

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.linear_means(h), self.linear_log_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, age, gender, height, weight, sbp, args):
        ED = x[:,0,...]

        if 'En' in args.mapping:
            label = self.mapping(torch.cat((age.float(), gender.float(),
                                            height.float(), weight.float(), sbp.float()), dim=-1))
            label = label.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
        else:
            age2 = age.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
            gender2 = gender.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
            height2 = height.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
            weight2 = weight.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
            sbp2 = sbp.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
            label = torch.cat((age2, gender2, height2, weight2, sbp2), dim=1)
        x = torch.cat((ED, label), 1)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z, age, gender, height, weight, sbp, args):

        if 'De' in args.mapping:
            label = self.mapping(
                torch.cat((age.float(), gender.float(), height.float(), weight.float(), sbp.float()), dim=-1))
        else:
            label = torch.cat((age, gender, height, weight, sbp), dim=-1)

        if 'after' in args.lstmpos:
            size_into_seq = args.z_dim
        elif 'De' in args.mapping:
            size_into_seq = args.z_dim + args.mapping_number
        else:
            size_into_seq = args.z_dim + args.condition
        x_all = []
        z_all = torch.empty((z.shape[0],20, size_into_seq)).cuda()
        # disp_all = torch.empty((z.shape[0], 20, 3, 64,128,128)).cuda()

        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        # t = 0, original z; t > 0, lstm z

        if args.lstmpos=='before':
            z = torch.cat((z, label), -1)

        for t in range(0, 20):
            if args.model =='lstm':
                if t == 0:
                    out = z.unsqueeze(1)
                    #out, (hidden_state,cell_state) = self.lstm(z.unsqueeze(1), (hidden_state, cell_state))
                else:
                    out, (hidden_state, cell_state) = self.lstm(out, (hidden_state, cell_state))
            if args.model == 'lstmcell':
                # for the first time step the input is the feature vector
                if t == 0:
                    ##debug
                    out = z
                    #out, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                    out = hidden_state

            if args.lstmpos == 'after':
                z_time = torch.cat((out.squeeze(1), label), -1)
            else:
                z_time = out.squeeze(1)
            h = self.fc2(z_time)
            x = self.decoder(h)
            x_all.append(x.unsqueeze(1))
            z_all[:,t]=out

        if args.visuallatent:
            return torch.cat(x_all, dim=1), z_all
        else:
            return torch.cat(x_all, dim=1)

    def forward(self, x, age, gender, height, weight, sbp, args):
        z, mu, logvar = self.encode(x, age, gender, height, weight, sbp, args)
        if args.visuallatent:
            z, z_all = self.decode(z, age, gender, height, weight, sbp, args)
            return z, mu, logvar, z_all
        else:
            z = self.decode(z, age, gender, height, weight, sbp, args)
            return z, mu, logvar

class GenCVAE_Seq_no_comment(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=8, nf=4, depth=64,label_num=4, condition = 5, args=None):
        super(GenCVAE_Seq_no_comment, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n, channel:4labels + 6 conditions
        if 'En' in args.mapping:
            self.conv1 = nn.Conv3d(label_num + args.mapping_number, nf, kernel_size=4, stride=2,
                                   padding=1)  # layer size increasing
        else:
            self.conv1 = nn.Conv3d(label_num+condition, nf, kernel_size=4, stride=2, padding=1)  # layer size increasing
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf * 8 * depth / 16 * img_size / 16 * img_size / 16)

        self.linear_means = nn.Linear(h_dim, z_dim)
        self.linear_log_var = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.fc_disp = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)
        if 'De' in args.mapping:
            self.fc2 = nn.Linear(z_dim + args.mapping_number, h_dim)
        else:
            self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.deconv1_dsp = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2_dsp = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3_dsp = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4_dsp = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16),
                      H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
        )
        self.decoder_disp = nn.Sequential(
            UnFlatten(C=int(nf * 8), D=int(depth / 16), H=int(img_size / 16), W=int(img_size / 16)),
            self.deconv1_dsp,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.deconv2_dsp,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.deconv3_dsp,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4_dsp,
            nn.Conv3d(label_num, 3, kernel_size=3, stride=1, padding=1)
        )
        if 'after' in args.lstmpos:
            self.input_size = z_dim
        elif 'De' in args.mapping:
            self.input_size = z_dim + args.mapping_number
        else:
            self.input_size = z_dim + args.condition
        # self.input_size = z_dim
        self.length_sequence = 20
        self.hidden_size = self.input_size
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            # dropout=0.2,
        )
        self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=True)
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        if args.mapping!='none':
            self.mapping = MLP(condition, args.mapping_number, 64, 8, weight_norm=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.linear_means(h), self.linear_log_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, age, gender, height, weight, sbp, args):
        ED = x[:,0,...]
        if 'En' in args.mapping:
            height1, weight1, sbp1 = height.unsqueeze(-1), weight.unsqueeze(-1), sbp.unsqueeze(-1)
            age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)
            label = self.mapping(torch.cat((age1.float(), gender1.float(),
                                            height1.float(), weight1.float(), sbp1.float()), dim=-1))
            label = label.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
        else:
            age2 = age.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            gender2 = gender.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            height2 = height.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            weight2 = weight.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            sbp2 = sbp.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            label = torch.cat((age2, gender2, height2, weight2, sbp2), dim=1)
        x = torch.cat((ED, label), dim=1)

        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z, age, gender, height, weight, sbp, args):

        height1, weight1, sbp1 = height.unsqueeze(-1), weight.unsqueeze(-1), sbp.unsqueeze(-1)
        age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)

        if 'De' in args.mapping:
            label = self.mapping(
                torch.cat((age1.float(), gender1.float(), height1.float(), weight1.float(), sbp1.float()), dim=-1))
        else:
            label = torch.cat((age1, gender1, height1, weight1, sbp1), dim=-1)

        if 'after' in args.lstmpos:
            size_into_seq = args.z_dim
        elif 'De' in args.mapping:
            size_into_seq = args.z_dim + args.mapping_number
        else:
            size_into_seq = args.z_dim + args.condition

        x_all = []
        # z_all = torch.empty((z.shape[0], 20, args.z_dim)).cuda()
        z_all = torch.empty((z.shape[0], 20, size_into_seq)).cuda()
        disp_all = torch.empty((z.shape[0], 20, 3, 64, 128, 128)).cuda()

        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        if args.lstmpos == 'before':
            z = torch.cat((z, label), -1)
        # t = 0, original z; t > 0, lstm z

        for t in range(0, 20):

                # for the first time step the input is the feature vector
            if t == 0:
                ##debug
                out = z
                #out, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                out = hidden_state

            if args.lstmpos == 'after':
                z_time = torch.cat((out.squeeze(1), label), -1)
            else:
                z_time = out.squeeze(1)
            h = self.fc2(z_time)
            x = self.decoder(h)
            x_all.append(x.unsqueeze(1))
            z_all[:, t] = out


        if args.visuallatent:
            return torch.cat(x_all, dim=1), disp_all, z_all
        else:
            return torch.cat(x_all, dim=1), disp_all

    def forward(self, x, age, gender, height, weight, sbp, args):
        z, mu, logvar = self.encode(x, age, gender, height, weight, sbp, args)
        if args.visuallatent:
            z, disp_all, z_all = self.decode(z, age, gender, height, weight, sbp, args)
            return z, disp_all, mu, logvar, z_all
        else:
            z, disp_all = self.decode(z, age, gender, height, weight, sbp, args)
            return z, disp_all, mu, logvar
# beta-conditional-VAE 3D for cardiac segmentation generation

class GenCVAE_Seq_UKBB_edes(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=8, nf=4, depth=64, label_num=4, condition=3, args=None):
        super(GenCVAE_Seq_UKBB_edes, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n, channel:4labels + 6 conditions
        if 'En' in args.mapping:
            self.conv1 = nn.Conv3d(label_num + args.mapping_number, nf, kernel_size=4, stride=2,
                                   padding=1)  # layer size increasing
        else:
            self.conv1 = nn.Conv3d(label_num+condition, nf, kernel_size=4, stride=2, padding=1)  # layer size increasing
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf * 8 * depth / 16 * img_size / 16 * img_size / 16)

        self.linear_means = nn.Linear(h_dim, z_dim)
        self.linear_log_var = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.fc_disp = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)
        if 'De' in args.mapping:
            self.fc2 = nn.Linear(z_dim + args.mapping_number, h_dim)
        else:
            self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.deconv1_dsp = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2_dsp = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3_dsp = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4_dsp = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16),
                      H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
        )
        self.decoder_disp = nn.Sequential(
            UnFlatten(C=int(nf * 8), D=int(depth / 16), H=int(img_size / 16), W=int(img_size / 16)),
            self.deconv1_dsp,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.deconv2_dsp,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.deconv3_dsp,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4_dsp,
            nn.Conv3d(label_num, 3, kernel_size=3, stride=1, padding=1)
        )
        if 'after' in args.lstmpos:
            self.input_size = z_dim
        elif 'De' in args.mapping:
            self.input_size = z_dim + args.mapping_number
        else:
            self.input_size = z_dim + args.condition
        # self.input_size = z_dim
        self.length_sequence = 50
        self.hidden_size = self.input_size
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            # dropout=0.2,
        )
        self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=True)
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        if args.mapping!='none':
            self.mapping = MLP(condition, args.mapping_number, 64, 8, weight_norm=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.linear_means(h), self.linear_log_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, age, gender, hypertension, args):
        ED = x[:,0,...]
        if 'En' in args.mapping:
            hypertension1 = hypertension.unsqueeze(-1)
            age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)
            label = self.mapping(torch.cat((age1.float(), gender1.float(),
                                            hypertension1.float()), dim=-1))
            label = label.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(3, 4, 0, 1, 2)
        else:
            age2 = age.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            gender2 = gender.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            hypertension2 = hypertension.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
            label = torch.cat((age2, gender2, hypertension2), dim=1)
        x = torch.cat((ED, label), dim=1)

        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z, age, gender, hypertension, args):

        hypertension1 = hypertension.unsqueeze(-1)
        age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)

        if 'De' in args.mapping:
            label = self.mapping(
                torch.cat((age1.float(), gender1.float(), hypertension1.float()), dim=-1))
        else:
            label = torch.cat((age1, gender1, hypertension1), dim=-1)

        if 'after' in args.lstmpos:
            size_into_seq = args.z_dim
        elif 'De' in args.mapping:
            size_into_seq = args.z_dim + args.mapping_number
        else:
            size_into_seq = args.z_dim + args.condition

        x_all = []
        # z_all = torch.empty((z.shape[0], 20, args.z_dim)).cuda()
        z_all = torch.empty((z.shape[0], self.length_sequence, size_into_seq)).cuda()
        disp_all = torch.empty((z.shape[0], self.length_sequence, 3, 64, 128, 128)).cuda()

        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        if args.lstmpos == 'before':
            z = torch.cat((z, label), -1)
        # t = 0, original z; t > 0, lstm z

        for t in range(0, self.length_sequence):

                # for the first time step the input is the feature vector
            if t == 0:
                ##debug
                out = z
                #out, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                out = hidden_state

            if args.lstmpos == 'after':
                z_time = torch.cat((out.squeeze(1), label), -1)
            else:
                z_time = out.squeeze(1)
            h = self.fc2(z_time)
            x = self.decoder(h)
            x_all.append(x.unsqueeze(1))
            z_all[:, t] = out


        if args.visuallatent:
            return torch.cat(x_all, dim=1), disp_all, z_all
        else:
            return torch.cat(x_all, dim=1), disp_all

    def forward(self, x, age, gender, hypertension, args):
        z, mu, logvar = self.encode(x, age, gender, hypertension, args)
        if args.visuallatent:
            z, disp_all, z_all = self.decode(z, age, gender, hypertension, args)
            return z, disp_all, mu, logvar, z_all
        else:
            z, disp_all = self.decode(z, age, gender, hypertension, args)
            return z, disp_all, mu, logvar
# beta-conditional-VAE 3D for cardiac segmentation generation


#4D volume generation
class GenCVAE_Seq_dsp_con(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=8, nf=4, depth=64,label_num=4, condition = 5):
        super(GenCVAE_Seq_dsp_con, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n, channel:4labels + 6 conditions
        self.conv1 = nn.Conv3d(label_num+condition, nf, kernel_size=4, stride=2, padding=1)  # layer size increasing
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf * 2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf * 8 * depth / 16 * img_size / 16 * img_size / 16)

        self.linear_means = nn.Linear(h_dim, z_dim)
        self.linear_log_var = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.fc_disp = nn.Linear(z_dim + condition, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)
        self.deconv1_dsp = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2_dsp = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3_dsp = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4_dsp = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16), H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
        )
        self.decoder_disp = nn.Sequential(
            UnFlatten(C=int(nf * 8), D=int(depth / 16), H=int(img_size / 16), W=int(img_size / 16)),
            self.deconv1_dsp,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.deconv2_dsp,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.deconv3_dsp,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4_dsp,
            nn.Conv3d(label_num, 3, kernel_size=3, stride=1, padding=1)
        )

        self.input_size = z_dim
        self.length_sequence = 20
        self.hidden_size = self.input_size
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            # dropout=0.2,
        )
        self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=True)
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.linear_means(h), self.linear_log_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, age, gender, height, weight, sbp, args):
        ED = x[ :, 0, ...]
        age2 = age.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        gender2 = gender.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        height2 = height.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        weight2 = weight.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        sbp2 = sbp.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        x = torch.cat((ED, age2, gender2, height2, weight2, sbp2), dim=1)

        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z, age, gender, height, weight, sbp, args):

        height1, weight1, sbp1 = height.unsqueeze(-1), weight.unsqueeze(-1), sbp.unsqueeze(-1)
        age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)
        x_all = []
        disp_all = torch.empty((z.shape[0], 20, 3, 64, 128, 128)).cuda()

        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        # t = 0, original z; t > 0, lstm z
        for t in range(0, 21):
            if args.model == 'gru':
                if t == 0:
                    out = z.unsqueeze(1)
                    #out, hidden_state = self.GRU_dec(z.unsqueeze(1), hidden_state)
                else:
                    out, hidden_state = self.GRU_dec(out, hidden_state)
            if args.model =='lstm':
                if t == 0:
                    out = z.unsqueeze(1)
                    #out, (hidden_state,cell_state) = self.lstm(z.unsqueeze(1), (hidden_state, cell_state))
                else:
                    out, (hidden_state, cell_state) = self.lstm(out, (hidden_state, cell_state))
            if args.model == 'grucell':
                # for the first time step the input is the feature vector
                if t == 0:
                    out = z
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state = self.gru_cell(z, hidden_state)
                    out = hidden_state
            if args.model == 'lstmcell':
                # for the first time step the input is the feature vector
                if t == 0:
                    ##debug
                    out = z
                    #out, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                    out = hidden_state


            if args.disp:
                if t ==0:
                    # distangment z contains condition information, but disp didn't contain temperol information
                    z_ED = torch.cat((out, age1, gender1, height1, weight1, sbp1), dim=-1)
                    h = self.fc2(z_ED)
                    x_ED = self.decoder(h)
                    #x_all.append(x_ED.unsqueeze(1))
                else:
                    disp = self.fc_disp(torch.cat((out.squeeze(1), age1, gender1, height1, weight1, sbp1), dim=-1))
                    disp = self.decoder_disp(disp)
                    disp_all[:, t-1, ...] = disp
                    #print(disp.min(), disp.max(), disp.mean(), disp.std())
                    x = warp(x_ED, disp, interp_mode='bilinear')
                    x_all.append(x.unsqueeze(1))
            else:
                z_time = torch.cat((out.squeeze(1), age1, gender1, height1, weight1, sbp1), dim=-1)
                h = self.fc2(z_time)
                x = self.decoder(h)
                x_all.append(x.unsqueeze(1))

        return torch.cat(x_all, dim=1), disp_all

    def forward(self, x, age, gender, height, weight, sbp, args):
        z, mu, logvar = self.encode(x, age, gender, height, weight, sbp, args)
        z, disp_all = self.decode(z, age, gender, height, weight, sbp, args)
        return z, disp_all, mu, logvar

class GenCVAE_Seq_forvaegan(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=8, nf=4, depth=64,label_num=4, condition = 5):
        super(GenCVAE_Seq_forvaegan, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n, channel:4labels + 6 conditions
        self.conv1 = nn.Conv3d(label_num+condition, nf, kernel_size=4, stride=2, padding=1)  # layer size increasing
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf * 8 * depth / 16 * img_size / 16 * img_size / 16)

        self.linear_means = nn.Linear(h_dim, z_dim)
        self.linear_log_var = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim + condition, h_dim)
        self.fc_disp = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)
        self.deconv1_dsp = nn.ConvTranspose3d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2_dsp = nn.ConvTranspose3d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3_dsp = nn.ConvTranspose3d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4_dsp = nn.ConvTranspose3d(nf, label_num, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16), H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
        )
        self.decoder_disp = nn.Sequential(
            UnFlatten(C=int(nf * 8), D=int(depth / 16), H=int(img_size / 16), W=int(img_size / 16)),
            self.deconv1_dsp,
            nn.BatchNorm3d(nf * 4),
            nn.LeakyReLU(0.2),
            self.deconv2_dsp,
            nn.BatchNorm3d(nf * 2),
            nn.LeakyReLU(0.2),
            self.deconv3_dsp,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4_dsp,
            nn.Conv3d(label_num, 3, kernel_size=3, stride=1, padding=1)
        )

        self.input_size = z_dim
        self.length_sequence = 20
        self.hidden_size = self.input_size
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True,
            # dropout=0.2,
        )
        self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=True)
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.linear_means(h), self.linear_log_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x, age, gender, height, weight, sbp, args):
        ED = x[:,0,...]
        age2 = age.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        gender2 = gender.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        height2 = height.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        weight2 = weight.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        sbp2 = sbp.repeat(x.shape[-3], x.shape[-1], x.shape[-2], 1, 1).permute(4, 3, 0, 1, 2)
        x = torch.cat((ED, age2, gender2, height2, weight2, sbp2), dim=1)

        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z, age, gender, height, weight, sbp, args):

        height1, weight1, sbp1 = height.unsqueeze(-1), weight.unsqueeze(-1), sbp.unsqueeze(-1)
        age1, gender1 = age.unsqueeze(-1), gender.unsqueeze(-1)
        x_all = []
        disp_all = torch.empty((z.shape[0], 20, 3, 64,128,128)).cuda()

        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        # t = 0, original z; t > 0, lstm z

        for t in range(0, 20):
            if args.model == 'gru':
                if t == 0:
                    out = z.unsqueeze(1)
                    #out, hidden_state = self.GRU_dec(z.unsqueeze(1), hidden_state)
                else:
                    out, hidden_state = self.GRU_dec(out, hidden_state)
            if args.model =='lstm':
                if t == 0:
                    out = z.unsqueeze(1)
                    #out, (hidden_state,cell_state) = self.lstm(z.unsqueeze(1), (hidden_state, cell_state))
                else:
                    out, (hidden_state, cell_state) = self.lstm(out, (hidden_state, cell_state))
            if args.model == 'grucell':
                # for the first time step the input is the feature vector
                if t == 0:
                    out = z
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state = self.gru_cell(z, hidden_state)
                    out = hidden_state
            if args.model == 'lstmcell':
                # for the first time step the input is the feature vector

                hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                out = hidden_state

            z_time = torch.cat((out.squeeze(1), age1, gender1, height1, weight1, sbp1), dim=-1)
            h = self.fc2(z_time)
            x = self.decoder(h)
            x_all.append(x.unsqueeze(1))

        return torch.cat(x_all, dim=1), disp_all

    def forward(self, x, age, gender, height, weight, sbp, args):
        z, mu, logvar = self.encode(x, age, gender, height, weight, sbp, args)
        z, disp_all = self.decode(z, age, gender, height, weight, sbp, args)
        return z, disp_all, mu, logvar
# beta-conditional-VAE 3D for cardiac segmentation generation

class GenCDCGAN_G_seq(nn.Module):
    """ cardiac segmentation DCGAN for GenScan """
    def __init__(self, z_dim=128, nf=4):
        super(GenCDCGAN_G_seq, self).__init__()
        self.z_dim = z_dim
        self.decon0 = nn.ConvTranspose3d(z_dim, nf * 8, kernel_size=(4, 8, 8))
        self.decon0_c = nn.ConvTranspose3d(5, nf*8, kernel_size=(4, 8, 8))
        self.deconv1 = nn.ConvTranspose3d(nf*8*2, nf*4*2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf*4*2, nf*2*2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf*2*2, nf*2, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf*2, 4, kernel_size=4, stride=2, padding=1)
        self.decoder = nn.Sequential(
            # self.decon0,
            self.deconv1,
            nn.BatchNorm3d(nf*4*2),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(nf*2),
            self.deconv4
        )
        self.input_size = z_dim
        self.length_sequence = 20
        self.hidden_size = self.input_size
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            # dropout=0.2,
        )
        self.GRU_dec = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, batch_first=True)
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # gru cell
        self.gru_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)


    def forward(self, z, age, gender, height, weight, sbp, args):
        age, gender, height, weight, sbp = age.view(-1, 1, 1, 1, 1), gender.view(-1, 1, 1, 1, 1), \
                                           height.view(-1, 1, 1, 1, 1), weight.view(-1, 1, 1, 1, 1), \
                                           sbp.view(-1, 1, 1, 1, 1)

        x_all = []
        #disp_all = torch.empty((z.shape[0], 20, 3, 64, 128, 128)).cuda()
        # one-to-many, build the output tensor
        if 'cell' in args.model:
            hidden_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(z.shape[0], self.hidden_size).cuda()
        else:
            hidden_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()
            cell_state = torch.zeros(self.num_layers, z.shape[0], self.hidden_size).cuda()

        # t = 0, original z; t > 0, lstm z
        c = self.decon0_c(torch.cat([age, gender, height, weight, sbp], 1))
        for t in range(0, 20):
            if args.model == 'gru':
                if t == 0:
                    out = z.unsqueeze(1)
                    # out, hidden_state = self.GRU_dec(z.unsqueeze(1), hidden_state)
                else:
                    out, hidden_state = self.GRU_dec(out, hidden_state)
            if args.model == 'lstm':
                if t == 0:
                    out = z.unsqueeze(1)
                    # out, (hidden_state,cell_state) = self.lstm(z.unsqueeze(1), (hidden_state, cell_state))
                else:
                    out, (hidden_state, cell_state) = self.lstm(out, (hidden_state, cell_state))
            if args.model == 'grucell':
                # for the first time step the input is the feature vector
                if t == 0:
                    out = z
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state = self.gru_cell(z, hidden_state)
                    out = hidden_state
            if args.model == 'lstmcell':
                # for the first time step the input is the feature vector
                if args.gan == 'seq20':
                    if t == 0:
                        ##debug
                        out = z
                    # # for the 2nd+ time step, using teacher forcer
                    else:
                        hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                        out = hidden_state
                else:
                    hidden_state, cell_state = self.lstm_cell(z, (hidden_state, cell_state))
                    out = hidden_state
            z_out = out.view(-1, self.z_dim, 1, 1, 1)
            z_out = self.decon0(z_out)
            z_out = torch.cat([z_out, c], 1)
            x = self.decoder(z_out)
            x_all.append(x.unsqueeze(1))

        return torch.cat(x_all, dim=1)

class GenCDCGAN_D_seq(nn.Module):

    def __init__(self, classnum=1, channel=4, args=None, nf=4):
        super(GenCDCGAN_D_seq, self).__init__()

        self.conv1 = nn.Conv3d(channel, nf, kernel_size=4, stride=2, padding=1)
        self.conv1_c = nn.Conv3d(5, nf, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv1_c_bn = nn.BatchNorm3d(nf)
        self.conv1_lr = nn.LeakyReLU(0.2)
        self.conv1_c_lr = nn.LeakyReLU(0.2)

        # size nf x 64/2 x n/2 x n/2
        if args.vaegan:# cvaegan: without condition in inpus of discriminator
            NF = nf
        else: # cgan: with condition in inputs of discriminator
            NF = nf*2
        self.conv2 = nn.Conv3d(NF, NF*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(NF*2, NF*4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(NF*4, NF*8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16
        self.conv5 = nn.Conv3d(NF*8, classnum, kernel_size=(4, 8, 8))
        # size 1 x 1 x 1 x 1

        self.encoder = nn.Sequential(
            #self.conv1,
            #nn.BatchNorm3d(nf),
            #nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(NF*2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(NF*4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(NF*8),
            nn.LeakyReLU(0.2),
            self.conv5,
            nn.Sigmoid()
        )

    def forward(self, x, age, gender, height, weight, sbp, args):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_lr(x)
        if not args.vaegan:
            age = age.repeat(64, 128, 128, 1, 1).permute(4, 3, 0, 1, 2)
            gender = gender.repeat(64, 128, 128, 1, 1).permute(4, 3, 0, 1, 2)
            height = height.repeat(64, 128, 128, 1, 1).permute(4, 3, 0, 1, 2)
            weight = weight.repeat(64, 128, 128, 1, 1).permute(4, 3, 0, 1, 2)
            sbp = sbp.repeat(64, 128, 128, 1, 1).permute(4, 3, 0, 1, 2)
            c = self.conv1_c(torch.cat([age, gender, height, weight, sbp], 1))
            c = self.conv1_c_bn(c)
            c = self.conv1_c_lr(c)
            x = torch.cat([x, c], 1)

        h = self.encoder(x).squeeze()
        return h

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, fc_dim, n_fc,
                 weight_norm=False, activation='relu', normalize_mlp=True):#, pixel_norm=False):
        super(MLP, self).__init__()
        # if weight_norm:
        #     linear = EqualLinear
        # else:
        #     linear = nn.Linear
        linear = nn.Linear
        if activation == 'lrelu':
            actvn = nn.LeakyReLU(0.2,True)
        # elif activation == 'blrelu':
        #     actvn = BidirectionalLeakyReLU()
        else:
            actvn = nn.ReLU(True)

        self.input_dim = input_dim
        self.model = []

        # normalize input
        if normalize_mlp:
            self.model += [PixelNorm()]

         # set the first layer
        self.model += [linear(input_dim, fc_dim),
                       actvn]
        if normalize_mlp:
            self.model += [PixelNorm()]

        # set the inner layers
        for i in range(n_fc - 2):
            self.model += [linear(fc_dim, fc_dim),
                           actvn]
            if normalize_mlp:
                self.model += [PixelNorm()]

        # set the last layer
        self.model += [linear(fc_dim, out_dim)] # no output activations

        # normalize output
        if normalize_mlp:
            self.model += [PixelNorm()]

        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        out = self.model(input)
        return out

class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)
