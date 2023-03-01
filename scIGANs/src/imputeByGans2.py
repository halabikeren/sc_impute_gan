## last update: 2019/04/28

from __future__ import print_function, division
import argparse
import os
import random

import numpy as np
import pandas as pd
import sys
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

OMP_NUM_THREADS=1
MAX_IMG_SIZE = 10 #(should be sqrt of default opt.latent_dim)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=20, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=100, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--dpt', type=str, default='', help='load discriminator model')
parser.add_argument('--gpt', type=str, default='', help='load generator model')
parser.add_argument('--train', help='train the network', action='store_true')
parser.add_argument('--impute', help='do imputation', action='store_true')
parser.add_argument('--sim_size', type=int, default=200, help='number of sim_imgs in each type')
parser.add_argument('--file_d', type=str, default='', help='path of data file')
parser.add_argument('--file_c', type=str, default='', help='path of cls file')
parser.add_argument('--file_t', type=str, default='', help='path of tech file')
parser.add_argument('--ct_ncls', type=int, default=4, help='number of cell type clusters')
parser.add_argument('--tech_ncls', type=int, default=4, help='number of technology clusters')
parser.add_argument('--knn_k', type=int, default=10, help='neighbors used')
parser.add_argument('--lr_rate', type=int, default=10, help='rate for slow learning')
parser.add_argument('--threshold', type=float, default=0.01, help='the convergence threshold')
parser.add_argument('--job_name', type=str, default="",
                    help='the user-defined job name, which will be used to name the output files.')
parser.add_argument('--outdir', type=str, default=".", help='the directory for output.')
parser.add_argument('--input_image', type=bool, default=True, help="indicator whether image data should be given as input to the generator and then the output will undergo dropout during training. if False, only noise will be given as input")
parser.add_argument('--dropout_shape', type=int, default=2, help='shape parameter for logit function on dropout values on which binomial distribution is applied')
parser.add_argument('--dropout_percentile', type=int, default=65, help='the percentile under which gene expression values are more likely to be dropped out')
parser.add_argument('--add_noise', type=bool, default=False, help='indicator to if noise should be added to the input of the generator or not')
parser.add_argument('--do_partition', type=bool, default=True, help='indicator weather partitioning of the data should be applied or not')
parser.add_argument('--partition_method', type=int, default=0, help='integer corresponding to partition method 0 for partitions without overlaps and without repeats, 1 for random partitions with repeats, 2 for partitions with overlaps, without repeats')
parser.add_argument('--partitions_nreps', type=int, default=5, help='integer corresponding the number of repeated partitions')
parser.add_argument('--partitions_overlap_size', type=int, default=100, help='number of overlapping genes between partitions')

opt = parser.parse_args()
max_ct_ncls = opt.ct_ncls  #
max_t_ncls = opt.tech_ncls  #
torch.set_num_threads(opt.n_cpu)
if opt.img_size > MAX_IMG_SIZE:
    opt.img_size = MAX_IMG_SIZE

job_name = opt.job_name
GANs_models = opt.outdir + '/GANs_models'
if (job_name == ""):
    job_name = os.path.basename(opt.file_d) + "-" + os.path.basename(opt.file_c) + "-" + os.path.basename(opt.file_b)
model_basename = job_name + "-" + str(opt.latent_dim) + "-" + str(opt.n_epochs) + "-" + str(opt.ct_ncls) + "-" + str(opt.tech_ncls)
if os.path.isdir(GANs_models) != True:
    os.makedirs(GANs_models)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def dropout_indicator(scData, shape=1, percentile=65):
    """
    This is similar to Splat package
    Input:
    scData can be the output of simulator or any refined version of it
    (e.g. with technical noise)
    shape: the shape of the logistic function
    percentile: the mid-point of logistic functions is set to the given percentile
    of the input scData
    returns: np.array containing binary indicators showing dropouts
    """
    scData = np.array(scData)
    scData_log = np.log(np.add(scData, 1))
    log_mid_point = np.percentile(scData_log, percentile)
    prob_ber = np.true_divide(1, 1 + np.exp(-1 * shape * (scData_log - log_mid_point)))

    binary_ind = np.random.binomial(n=1, p=prob_ber)

    return binary_ind


def convert_to_UMIcounts(scData):
    """
    Input: scData can be the output of simulator or any refined version of it
    (e.g. with technical noise)
    """
    return np.random.poisson(scData)

# %% for debug use only
# opt.file_d='ercc.csv'
# opt.file_c='ercc.label.txt'
# opt.img_size=9
# opt.train=True
# opt.n_epochs = 1
# cuda = False
# %%

class MyPartitions:

    def __init__(self, d_file: str, cls_file: str, tech_file: str, partition_method: int = 0, nrepeats: int = 5, overlap_size: int = 20, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            partition_method: 0 for partitions without overlaps and without repeats
                              1 for random partitions with repeats
                              2 for partitions with overlaps, without repeats
        """
        full_data = pd.read_csv(d_file, index_col=False).drop(["Unnamed: 0"], axis=1)
        full_ct_labels = pd.Categorical(pd.read_csv(cls_file, header=None, index_col=False).iloc[:,0]).codes
        full_tech_labels = pd.Categorical(pd.read_csv(tech_file, header=None, index_col=False).iloc[:,0]).codes

        self.partitions = []
        if not opt.do_partition:
            all_gene_indices = full_data.index.tolist()
            dataset = MyDataset(data=full_data, ct_labels=full_ct_labels, tech_labels=full_tech_labels, transform=transform)
            self.partitions.append((dataset, all_gene_indices))
        else:
            n_repeats = 1 if partition_method == 0 else nrepeats
            n_genes = full_data.shape[0]
            partition_jump = opt.img_size**2
            partition_size = opt.img_size**2
            if partition_method == 2:
                partition_jump -= overlap_size
            for rep in range(n_repeats):
                partition = []
                shuffled_gene_indices = full_data.index.tolist()
                if partition_method == 1:
                    random.shuffle(shuffled_gene_indices)
                full_data_to_partition = full_data.iloc[shuffled_gene_indices]
                covered_indices = []
                for i in range(0, n_genes, partition_jump):
                    data = full_data_to_partition.iloc[i:i+partition_size, :]
                    gene_indices = data.index.tolist()
                    covered_indices += gene_indices
                    dataset = MyDataset(data=data, ct_labels=full_ct_labels, tech_labels=full_tech_labels, transform=transform)
                    partition.append((dataset, gene_indices))
                assert(len(set(covered_indices)) == len(set(shuffled_gene_indices)))
                print(f"created {len(partition):,} partitions for repeat {rep}...")
                self.partitions.append(partition)

    def __len__(self):
        return len(self.partitions)



class MyDataset(Dataset):
    """Operations with the datasets."""

    def __init__(self, data, ct_labels, tech_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.data_cls = ct_labels
        self.data_technology = tech_labels
        self.transform = transform
        self.fig_h = opt.img_size  ##

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'double')  #
        ct_label = np.array(self.data_cls[idx]).astype('int32')  #
        tech_label = np.array(self.data_technology[idx]).astype('int32')  #
        sample = {'data': data, 'cell_type_label': ct_label, 'technology_label': tech_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, ct_label, tech_label = sample['data'], sample['cell_type_label'], sample['technology_label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))

        return {'data': torch.from_numpy(data),
                'cell_type_label': torch.from_numpy(ct_label),
                'technology_label': torch.from_numpy(tech_label)
                }


def one_hot(batch, depth):
    ones = torch.eye(depth)
    return ones.index_select(0, batch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.cn1 = 32
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (self.init_size ** 2)))
        self.l1p = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (opt.img_size ** 2)))

        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            #            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )

        self.conv_blocks_02p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=opt.img_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ct_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_03p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=opt.img_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_t_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(48, 0.8),
            nn.Conv2d(48, self.cn1, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
            nn.Sigmoid()
        )

    def forward(self, x, ct_label, t_label):
        out = self.l1p(x)
        out = out.view(out.shape[0], self.cn1, opt.img_size, opt.img_size)
        out01 = self.conv_blocks_01p(out)  # ([4, 32, 124, 124])
        #
        ct_label = ct_label.unsqueeze(2)
        ct_label = ct_label.unsqueeze(2)
        out02 = self.conv_blocks_02p(ct_label)  # ([4, 8, 124, 124])

        t_label = t_label.unsqueeze(2)
        t_label = t_label.unsqueeze(2)
        out03 = self.conv_blocks_03p(t_label)  # ([4, 8, 124, 124])

        out1 = torch.cat((out01, out02, out03), 1)
        out1 = self.conv_blocks_1(out1)
        return out1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32
        # pre
        self.pre = nn.Sequential(
            nn.Linear(opt.img_size ** 2, self.down_size0 ** 2),
        )

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1 // 2, 3, 1, 1),
            #            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(),
        )

        self.conv_blocks02p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=self.down_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ct_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )
        
        self.conv_blocks03p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=self.down_size),  # torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_t_ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        # Fully-connected layers

        down_dim = 32 * (self.down_size) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(),
        )
        # Upsampling 32X32
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, ct_label, t_label):
        out00 = self.pre(img.view((img.size()[0], -1))).view((img.size()[0], 1, self.down_size0, self.down_size0))
        out01 = self.down(out00)  # ([4, 16, 32, 32])

        ct_label = ct_label.unsqueeze(2)
        ct_label = ct_label.unsqueeze(2)
        out02 = self.conv_blocks02p(ct_label)  # ([4, 16, 32, 32])

        t_label = t_label.unsqueeze(2)
        t_label = t_label.unsqueeze(2)
        out03 = self.conv_blocks03p(t_label)  # ([4, 16, 32, 32])
        ####
        out1 = torch.cat((out01, out02, out03), 1)
        ######
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 32, self.down_size, self.down_size))
        return out


# %%
def my_knn_type(data_imp_org_k, sim_out_k, knn_k=10):
    sim_size = sim_out_k.shape[0]
    out = data_imp_org_k.copy()
    q1k = data_imp_org_k.reshape((opt.img_size * opt.img_size, 1))
    q1kl = np.int8(q1k > 0)  # get which part in cell k is >0
    q1kn = np.repeat(q1k * q1kl, repeats=sim_size, axis=1)  # get >0 part of cell k
    sim_out_tmp = sim_out_k.reshape((sim_size, opt.img_size * opt.img_size)).T
    sim_outn = sim_out_tmp * np.repeat(q1kl, repeats=sim_size, axis=1)  # get the >0 part of simmed ones
    diff = q1kn - sim_outn  # distance of cell k to simmed ones
    diff = diff * diff
    rel = np.sum(diff, axis=0)
    locs = np.where(q1kl == 0)[0]
    #        locs1 = np.where(q1kl==1)[0]
    sim_out_c = np.median(sim_out_tmp[:, rel.argsort()[0:knn_k]], axis=1)
    out[locs] = sim_out_c[locs]
    return out


# %%
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    print("scIGANs is running on GPUs.")
else:
    print("scIGANs is running on CPUs.")
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
transformed_datasets_partitions = MyPartitions(d_file=opt.file_d,
                                               cls_file=opt.file_c,
                                               tech_file=opt.file_t,
                                               partition_method=opt.partition_method,
                                               nrepeats=opt.partitions_nreps,
                                               overlap_size=opt.partitions_overlap_size,
                                               transform=transforms.Compose([
                                                    ToTensor()
                                                ]))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# %%
# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = opt.gamma
lambda_k = 0.001
k = opt.kt

nreps = len(transformed_datasets_partitions)

if opt.train:
    model_exists = os.path.isfile(GANs_models + '/' + model_basename + '-g.pt')
    if model_exists:
        overwrite = input(
            "WARNING: A trained model exists with the same settings for your data.\n         Do you want to train and overwrite it?: (y/n)\n")
        if overwrite != "y":
            print("The training was deprecated since optical model exists.")
            print("scIGANs continues imputation using existing model...")
            sys.exit()  # if model exists and do not want to train again, exit the program
    print(
        "The optimal model will be output in \"" + os.getcwd() + "/" + GANs_models + "\" with basename = " + model_basename)
    #    if opt.dpt!='' and cuda==True:
    #        discriminator.load_state_dict(torch.load(opt.dpt))
    #        generator.load_state_dict(torch.load(opt.gpt))
    #    if opt.dpt!='' and cuda != True:
    #        discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))
    #        generator.load_state_dict(torch.load(opt.gpt, map_location=lambda storage, loc: storage))
    max_M = sys.float_info.max
    min_dM = 0.001
    dM = 1
    z_vals = []
    for epoch in range(opt.n_epochs):
        cur_M = 0
        cur_dM = 1
        for rep in range(nreps):
            num_partitions = len(transformed_datasets_partitions.partitions[rep])
            for part in range(num_partitions):
                (transformed_dataset, gene_indices) = transformed_datasets_partitions.partitions[rep][part]
                dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=0, drop_last=True)
                for i, batch_sample in enumerate(dataloader):
                    imgs = batch_sample['data'].type(Tensor)
                    cell_type_label = batch_sample['cell_type_label']
                    technology_label = batch_sample['technology_label']
                    ct_label_oh = one_hot((cell_type_label).type(torch.LongTensor), max_ct_ncls).type(Tensor)  #
                    t_label_oh = one_hot((technology_label).type(torch.LongTensor), max_t_ncls).type(Tensor)  #

                    # Configure input
                    real_imgs = Variable(imgs.type(Tensor))

                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Sample noise as generator input - in case of partitioning, subsample size should be 100%
                    if opt.input_image and opt.do_partition and len(imgs.flatten()) < imgs.shape[0]*opt.latent_dim:
                        raise ValueError("image size larger than noise size despite of pairtiotioning")
                    z_orig = np.random.choice(a=imgs.flatten(), size=imgs.shape[0]*opt.latent_dim).reshape(imgs.shape[0], opt.latent_dim)
                    z_noise = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
                    if not opt.input_image:
                        z = Variable(Tensor(z_noise))
                    elif opt.input_image and not opt.add_noise:
                        z = Variable(Tensor(z_orig))
                    elif opt.input_image and  opt.add_noise:
                        z_noise = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
                        z = Variable(Tensor(z_orig+z_noise))

                    # Generate a batch of images
                    gen_imgs = generator(z, ct_label_oh, t_label_oh)

                    # add dropouts in case if input_image:
                    if opt.input_image:
                        transformed_gen_imgs = gen_imgs.reshape((-1, opt.img_size * opt.img_size)).detach().numpy().T
                        binary_ind = dropout_indicator(transformed_gen_imgs, opt.dropout_shape, opt.dropout_percentile) # ? gen_imgs is not s matrix with rows representing genes and columns representing cells
                        expr_O_L_D = np.multiply(binary_ind, transformed_gen_imgs)
                        stacked_expr_O_L_D = np.asarray([expr_O_L_D[:, idx][0:(opt.img_size * opt.img_size), ].reshape(1, opt.img_size, opt.img_size).astype('double') for idx in range(expr_O_L_D.shape[1])])
                        dropped_gen_imgs = torch.Tensor(stacked_expr_O_L_D)
                    else:
                        dropped_gen_imgs = gen_imgs
                    # Loss measures generator's ability to fool the discriminator
                    disc_on_gen_imgs = torch.abs(discriminator(dropped_gen_imgs, ct_label_oh, t_label_oh))
                    g_loss = torch.mean(disc_on_gen_imgs - dropped_gen_imgs)

                    g_loss.backward()
                    optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()

                    # Measure discriminator's ability to classify real from generated samples
                    d_real = discriminator(real_imgs, ct_label_oh, t_label_oh)
                    d_fake = discriminator(dropped_gen_imgs.detach(), ct_label_oh, t_label_oh)

                    d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
                    d_loss_fake = torch.mean(torch.abs(d_fake - dropped_gen_imgs.detach()))
                    d_loss = d_loss_real - k * d_loss_fake

                    d_loss.backward()
                    optimizer_D.step()

                    # ----------------
                    # Update weights
                    # ----------------

                    diff = torch.mean(gamma * d_loss_real - d_loss_fake)

                    # Update weight term for fake samples
                    k = k + lambda_k * np.asscalar(diff.detach().data.cpu().numpy())
                    k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                    # Update convergence metric
                    M = (d_loss_real + torch.abs(diff)).item()
                    cur_M += M
                    # --------------
                    # Log Progress
                    # --------------

                    sys.stdout.write("\r[Epoch %d/%d] [Repeat %d/%d] [Partition %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (
                    epoch + 1, opt.n_epochs, rep+1, nreps, part+1, num_partitions, i + 1, len(dataloader),
                    np.asscalar(d_loss.detach().data.cpu().numpy()), np.asscalar(g_loss.detach().data.cpu().numpy()),
                    M, dM, k))
                    sys.stdout.flush()
                    batches_done = epoch * len(dataloader) + i
                # get the M of current epoch
                cur_M = cur_M / len(dataloader)
                if cur_M < max_M:  # if current model is better than previous one
                    torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
                    torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
                    dM = min(max_M - cur_M, cur_M)
                    if dM < min_dM:  # if convergence threshold meets, stop training
                        print(
                            "Training was stopped after " + str(epoch + 1) + " epochs since the convergence threshold (" + str(
                                min_dM) + ".) reached: " + str(dM))
                        break
                    cur_dM = max_M - cur_M
                    max_M = cur_M
                if epoch + 1 == opt.n_epochs and cur_dM > min_dM:
                    print("Training was stopped after " + str(epoch + 1) + " epochs since the maximum epochs reached: " + str(
                        opt.n_epochs) + ".")
                    print("WARNING: the convergence threshold (" + str(min_dM) + ") was not met. Current value is: " + str(
                        cur_dM))
                    print("You may need more epochs to get the most optimal model!!!")

if opt.impute:
    imputed_datasets = []
    if opt.gpt == '':
        model_g = GANs_models + '/' + model_basename + '-g.pt'
        model_exists = os.path.isfile(model_g)
        if not model_exists:
            print("ERROR: There is no model exists with the given settings for your data.")
            print("Please set --train instead of --impute to train a model first.")
            sys.exit("scIGANs stopped!!!")  # if model exists and do not want to train again, exit the program
            print()
    else:
        model_g = opt.gpt
    print(model_g + " is used for imputation.")
    if cuda == True:
        # discriminator.load_state_dict(torch.load(opt.dpt))
        generator.load_state_dict(torch.load(model_g))
    else:
        # discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))
        generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))
    #############################################################
    ###        impute by cell type and technology type        ###
    #############################################################

    for rep in range(len(transformed_datasets_partitions.partitions)):
        imputed_data = []
        for (transformed_dataset, gene_indices) in transformed_datasets_partitions.partitions[rep]:
            data_imp_org = np.asarray(
                [transformed_dataset[i]['data'].numpy().reshape((opt.img_size * opt.img_size)) for i in range(len(transformed_dataset))]).T
            data_imp = data_imp_org.copy()
            sim_size = opt.sim_size
            sim_out = list()
            for i in range(opt.ct_ncls):
              ct_label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ct_ncls).type(Tensor)
              sim_out.append(list())
              for j in range(opt.tech_ncls):
                  t_label_oh = one_hot(torch.from_numpy(np.repeat(j, sim_size)).type(torch.LongTensor), max_t_ncls).type(Tensor)

                  # Sample noise as generator input
                  if opt.input_image and opt.do_partition and len(imgs.flatten()) < imgs.shape[0] * opt.latent_dim:
                      raise ValueError("image size larger than noise size despite of pairtiotioning")
                  z_orig = np.random.choice(a=data_imp_org.flatten(), size=sim_size * opt.latent_dim).reshape(
                      sim_size, opt.latent_dim)
                  z_noise = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
                  if not opt.input_image:
                      z = Variable(Tensor(z_noise))
                  elif opt.input_image and not opt.add_noise:
                      z = Variable(Tensor(z_orig))
                  elif opt.input_image and opt.add_noise:
                      z_noise = np.random.normal(0, 1, (sim_size, opt.latent_dim))
                      z = Variable(Tensor(z_orig + z_noise))

                  # Generate a batch of images
                  fake_imgs = generator(z, ct_label_oh, t_label_oh).detach().data.cpu().numpy()
                  sim_out[i].append(fake_imgs)

            # by type
            sim_out_org = sim_out
            rels = np.asarray([my_knn_type(data_imp_org[:, k], sim_out_org[int(transformed_dataset[k]['cell_type_label']) - 1][int(transformed_dataset[k]['technology_label']) - 1], knn_k=opt.knn_k) for k in range(len(transformed_dataset))]).transpose()
            rels_df = pd.DataFrame(rels)
            rels_df.index = gene_indices
            imputed_data.append(rels_df)
        imputed_data = pd.concat(imputed_data).sort_index()
        imputed_datasets.append(imputed_data)

    full_imputed_data = pd.concat(imputed_datasets)
    full_imputed_data = full_imputed_data.reset_index()
    full_imputed_data = full_imputed_data.groupby('index').mean()
    full_imputed_data = full_imputed_data.transpose()
    full_imputed_data.to_csv(os.path.dirname(os.path.abspath(opt.file_d)) + '/scIGANs-' + job_name + '.csv')  # imputed data

