import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.rnn
from tqdm import tqdm
import torch_geometric.data
import graphnet
import glob
import os
import pickle
from sklearn import neighbors

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

        with open(f'test.pk', 'rb') as filehandle:
            self.tau_all = pickle.load(filehandle)
            self.S_all = pickle.load(filehandle)
            self.J_all = pickle.load(filehandle)
            self.L_all = pickle.load(filehandle)
            self.eps_all = pickle.load(filehandle)
            self.ratio_all = pickle.load(filehandle)
        
        self.n_training = len(self.tau_all)

        self.edge_index = [None] * self.n_training
        self.nodes = [None] * self.n_training
        self.edges = [None] * self.n_training
        self.u = [None] * self.n_training
        self.target = [None] * self.n_training

        for i in tqdm(range(self.n_training)):

            num_nodes = len(self.tau_all[i][0, :])
            index_tau = np.zeros((num_nodes, 1))
            
            index_tau[:, 0] = np.log10(self.tau_all[i][0, :])

            # Build the KDTree
            self.tree = neighbors.KDTree(index_tau)

            # Get neighbors
            receivers_list = self.tree.query_radius(index_tau, r=0.15)

            senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
            receivers = np.concatenate(receivers_list, axis=0)

            # Mask self edges
            mask = senders != receivers

            senders = torch.tensor(senders[mask].astype('long'))
            receivers = torch.tensor(receivers[mask].astype('long'))

            self.edge_index[i] = torch.cat([senders[None, :], receivers[None, :]], dim=0)

            n_edges = self.edge_index[i].shape[1]

            self.nodes[i] = np.zeros((num_nodes, 1))
            self.nodes[i][:, 0] = np.log10(self.S_all[i])

            self.edges[i] = np.zeros((n_edges, 1))
            tau0 = self.tau_all[i][0, self.edge_index[i][0, :]]
            tau1 = self.tau_all[i][0, self.edge_index[i][1, :]]
            self.edges[i][:, 0] = np.log10(np.abs(tau0 - tau1))

                        
            self.u[i] = np.zeros((1, 2))
            self.u[i][0, :] = np.array([np.log10(self.eps_all[i][0, 0]), np.log10(self.ratio_all[i][0, 0])], dtype=np.float32)
            
            self.target[i] = np.zeros((num_nodes, 1))
            self.target[i][:, 0] = np.log10(self.J_all[i])


            self.nodes[i] = torch.tensor(self.nodes[i].astype('float32'))
            self.edges[i] = torch.tensor(self.edges[i].astype('float32'))
            self.u[i] = torch.tensor(self.u[i].astype('float32'))
            self.target[i] = torch.tensor(self.target[i].astype('float32'))

        
    def __getitem__(self, index):
        node = self.nodes[index]
        edge_attr = self.edges[index]
        target = self.target[index]
        u = self.u[index]
        edge_index = self.edge_index[index]

        data = torch_geometric.data.Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=target, u=u)
        
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
        
        self.model = graphnet.EncodeProcessDecode(**hyper).to(self.device)        
                        
        self.model.load_state_dict(checkpoint['state_dict'])  

        self.dataset = Dataset()
        self.loader = torch_geometric.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
        
    def test(self):
        self.model.eval()
        loss_avg = 0
        t = tqdm(self.loader)
        correct = 0
        total = 0
        n = 1

        self.target = []
        self.out = []
        self.eps = []
        self.ratio = []
        
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):
                                
                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch
                
                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)
                
                out = self.model(node, edge_attr, edge_index, u, batch)

                n = len(data.ptr) - 1
                for i in range(n):
                    left = data.ptr[i]
                    right = data.ptr[i+1]                    
                    self.out.append(out[left:right, :].cpu().numpy())
                    self.target.append(target[left:right, :].cpu().numpy())
                    self.eps.append(data.u[i, 0].cpu().numpy())
                    self.ratio.append(data.u[i, 1].cpu().numpy())

        fig, ax = pl.subplots(nrows=5, ncols=5, figsize=(15,10), sharex='col')
        for i in range(25):
            ax.flat[i].semilogx(self.dataset.tau_all[i][0, :], self.target[i])
            ax.flat[i].semilogx(self.dataset.tau_all[i][0, :], self.out[i])
            ax.flat[i].text(0.05, 0.85, f'l$\epsilon$={self.eps[i]:5.3f}', transform=ax.flat[i].transAxes)
        
        fig.supxlabel(r'$\tau$')
        fig.supylabel('J')

        pl.savefig('2level.png')

if (__name__ == '__main__'):
    pl.close('all')
    network = Formal(batch_size=40, gpu=0, checkpoint=None)    
    network.test()
