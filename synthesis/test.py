import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
from tqdm import tqdm
import torch_geometric.data
import model
import glob
import os
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

class Formal(object):
    def __init__(self, batch_size=64, gpu=0, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}        
        
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size                           

        if (checkpoint is None):
            files = glob.glob('weights/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)

        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        
        hyper = checkpoint['hyperparameters']
        
        self.model = model.EncodeProcessDecode(**hyper).to(self.device)        
                        
        self.model.load_state_dict(checkpoint['state_dict'])  

        self.dataset = Dataset(n_training=40)
        self.loader = torch_geometric.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
        
    def test(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.loader)
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

            self.out = out.cpu().numpy()
            self.target = target.cpu().numpy()

if (__name__ == '__main__'):
    pl.close('all')
    network = Formal(batch_size=40, gpu=0, checkpoint=None)    
    network.test()
