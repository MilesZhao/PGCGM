import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.crystal_block = nn.Sequential(
                nn.Conv2d(3,16,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(16,32,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32,64,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64,96,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(96,128,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128,192,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(192,256,2,stride=1),
                nn.LeakyReLU(0.2),

                nn.Flatten()
            )

        self.sp_block = nn.Sequential(
                nn.Conv2d(192, 64, 2, 1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, 2, 1),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, 2, 1),
                nn.LeakyReLU(0.2),

                nn.Flatten()
            )

        self.dense_block = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, 1)
            )

    def forward(self, crystal, symm_mat):
        x1 = self.crystal_block(crystal)
        x2 = self.sp_block(symm_mat)
        x = torch.cat((x1, x2), 1)
        x = self.dense_block(x)

        return x

class Generator(nn.Module):
    def __init__(self, ele_vec_dim=23, noise_dim=128):
        super(Generator, self).__init__()
        self.sp_block = nn.Sequential(
                nn.Conv2d(192, 64, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 128, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 256, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Flatten()
            )

        self.ele_block = nn.Sequential(
                nn.Conv1d(ele_vec_dim, 64, 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Conv1d(64, 128, 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Flatten(),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

            )

        self.noise_block = nn.Sequential(
                nn.Linear(noise_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )

        self.coords_block1 = nn.Sequential(
                nn.ConvTranspose2d(512,1024,(2,2),(1,1)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.ConvTranspose2d(1024,512,(2,2),(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.ConvTranspose2d(512,256,(1,1),(1,1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.ConvTranspose2d(256,128,(1,1),(1,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.ConvTranspose2d(128,64,(1,1),(1,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64,3,(1,1),(1,1)),
                nn.Tanh()
            )

        self.length_block = nn.Sequential(
                nn.Linear(512,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64,32),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                nn.Linear(32,16),
                nn.BatchNorm1d(16),
                nn.ReLU(),

                nn.Linear(16,3),
                nn.Tanh()
            )

    def forward(self, sp_inputs, ele_inputs, z):
        sp_embedding = self.sp_block(sp_inputs)
        ele_embedding = self.ele_block(ele_inputs)
        z_embedding = self.noise_block(z)
        
        x1 = torch.cat((z_embedding, ele_embedding), 1)
        x2 = torch.cat((z_embedding, sp_embedding), 1)
        
        coords = self.coords_block1(x1.view(-1,512,1,1))
        length = self.length_block(x2)
        
        return coords,length

def calc_grad_penalty(netD, device, real_data, fake_data):
    real_mat, real_symm = real_data
    fake_mat, fake_symm = fake_data
    batch_size = real_mat.shape[0]

    alpha = torch.normal(0.0, 1.0, size=(batch_size,1,1,1)).to(device)
    interpolated_mat = alpha * real_mat + ((1 - alpha) * fake_mat)
    interpolated_mat = interpolated_mat.to(device)
    interpolated_mat = autograd.Variable(interpolated_mat, requires_grad=True)

    alpha = torch.normal(0.0, 1.0, size=(batch_size,1,1,1)).to(device)
    interpolated_symm = alpha * real_symm + ((1 - alpha) * fake_symm)
    interpolated_symm = interpolated_symm.to(device)
    interpolated_symm = autograd.Variable(interpolated_symm, requires_grad=True)

    pred_interpolates = netD(interpolated_mat, interpolated_symm)

    gradients = autograd.grad(
        outputs=pred_interpolates,
        inputs=[interpolated_mat, interpolated_symm],
        grad_outputs=torch.ones(pred_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    
    gradients0 = gradients[0].contiguous().view(batch_size, -1)
    gradients1 = gradients[1].contiguous().view(batch_size, -1)
    gradient_penalty = ((gradients0.norm(2, dim=1) - 1) ** 2).mean() + ((gradients1.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty