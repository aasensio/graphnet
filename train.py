import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
import torch_geometric.data
import time
from tqdm import tqdm
import model
import argparse
from sklearn import neighbors
import shortcar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_training=None):
        super(Dataset, self).__init__()

        # Define indices of the positions. The graph will connect all points at certain distance
        # in index, not in tau
        num_nodes = 30
        self.index_tau = np.zeros((num_nodes, 1))
        self.index_tau[:, 0] = np.arange(num_nodes)

        # Build the KDTree
        self.tree = neighbors.KDTree(self.index_tau)

        # Get neighbors
        receivers_list = self.tree.query_radius(self.index_tau, r=2)

        senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
        receivers = np.concatenate(receivers_list, axis=0)

        # Mask self edges
        mask = senders != receivers

        senders = torch.tensor(senders[mask].astype('long'))
        receivers = torch.tensor(receivers[mask].astype('long'))

        self.edge_index = torch.cat([senders[None, :], receivers[None, :]], dim=0)

        edge_index = self.edge_index.numpy()

        n_edges = self.edge_index.shape[1]

        self.dtau = np.zeros((n_training, n_edges, 1), dtype=np.float32)
        self.S = np.zeros((n_training, num_nodes, 1), dtype=np.float32)
        self.I = np.zeros((n_training, num_nodes), dtype=np.float32)
        self.I0 = np.zeros((n_training, 1), dtype=np.float32)
        
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel)

        log_tau_min = -8.0
        
        for i in tqdm(range(n_training)):
            delta_logtau = np.random.uniform(low=0.05, high=0.1)            
            
            logtau = log_tau_min + delta_logtau * self.index_tau[:, 0]
            tau = 10.0**logtau

            S = gp.sample_y(logtau[:, np.newaxis], 1)[:, 0]
            S -= np.min(S)

            self.S[i, :, 0] = S

            self.I0[i, 0] = np.random.uniform(low=-10, high=0.0)

            sol = shortcar.shortcar(tau, S, I0=10.0**self.I0[i, 0])

            self.dtau[i, :, 0] = np.log10(np.abs(tau[edge_index[0, :]] - tau[edge_index[1, :]]))            
            self.I[i, :] = np.log10(sol)

        self.dtau = torch.tensor(self.dtau)
        self.S = torch.tensor(self.S)
        self.I = torch.tensor(self.I)
        self.I0 = torch.tensor(self.I0)

        self.n_training = n_training               
        
    def __getitem__(self, index):
        node = self.S[index, :, :]
        edge_attr = self.dtau[index, :, :]
        target = self.I[index, :]
        u = self.I0[index, :]

        data = torch_geometric.data.Data(x=node, edge_index=self.edge_index, edge_attr=edge_attr, y=target, u=u)
        
        return data

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Formal(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, validation_split=0.2):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        self.device = 'cpu'

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size                

        # Neural network                
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}        

        self.hyperparameters = {
            'node_input_size': 1,
            'edge_input_size': 1,
            'global_input_size': 1, 
            'latent_size': 16,
            'mlp_hidden_size': 16,
            'mlp_n_hidden_layers': 2,
            'n_message_passing_steps': 5,
            'output_size': 1,
        }
        
        self.model = model.EncodeProcessDecode(**self.hyperparameters).to(self.device)

        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
    
        self.dataset = Dataset(n_training=1000)

        # Training set
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

         # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        self.train_loader = torch_geometric.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch_geometric.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

    def optimize(self, epochs, lr=3e-4):        

        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f'weights/{current_time}.pth'

        print(' Model: {0}'.format(self.out_name))
        
        # Copy model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.loss_fn = nn.MSELoss()

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            if  (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': self.hyperparameters,                    
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}')

            # self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0

        for batch_idx, (data) in enumerate(t):

            node = data.x
            edge_attr = data.edge_attr
            edge_index = data.edge_index
            target = data.y
            u = data.u.unsqueeze(-1)
            batch = data.batch

            node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
            u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)
                        
            self.optimizer.zero_grad()
            
            out = self.model(node, edge_attr, edge_index, u, batch)

            loss = self.loss_fn(out.squeeze(), target.squeeze())

            loss.backward()
            
            self.optimizer.step()
            
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        correct = 0
        total = 0
        n = 1
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):
                                
                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u.unsqueeze(-1)
                batch = data.batch

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)
                
                out = self.model(node, edge_attr, edge_index, u, batch)

                loss = self.loss_fn(out.squeeze(), target.squeeze())
                
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                                
                t.set_postfix(loss=loss_avg)            
   
        return loss_avg

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=8, type=int,
                    metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.08, type=float,
                    metavar='SPLIT', help='Validation split')
    
    
    parsed = vars(parser.parse_args())

    network = Formal(
            batch_size=parsed['batch'], 
            gpu=parsed['gpu'], 
            validation_split=parsed['split'], 
            smooth=parsed['smooth'])

    network.optimize(parsed['epochs'], lr=parsed['lr'])
