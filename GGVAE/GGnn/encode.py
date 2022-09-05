import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import Set2Set
from dgl.nn.pytorch.conv import NNConv

from GGVAE.my_dataset.graphs import GraphMol


class AtomEncoder(nn.Module):
    def __init__(self, n_atom_feat, n_bond_feat,
                 h_atom, n_bond, n_layers=2):
        super(AtomEncoder, self).__init__()
        self.embedding_atom = nn.Linear(n_atom_feat, h_atom)
        self.embedding_bond = nn.Linear(n_bond_feat, n_bond)
        self.mpnn = MPNN(h_atom, n_bond, n_layers)

    def forward(self, g: dgl.DGLGraph):
        h_atom = self.embedding_atom(g.ndata['n_feat'])
        h_bond = self.embedding_bond(g.edata['e_feat'])
        h_atom = self.mpnn(g, h_atom, h_bond)
        g.ndata['h'] = h_atom

        return dgl.mean_nodes(g, 'h')


class FragEncoder(nn.Module):
    def __init__(self, n_atom_feat, n_bond_feat, h_atom, h_bond,
                 n_frag_feat, n_fbond_feat, h_frag, h_fbond,
                 n_layers=2, latent_dim=1024):
        super(FragEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.atom_encoder = AtomEncoder(n_atom_feat, n_bond_feat,
                                        h_atom, h_bond, n_layers)
        self.embedding_frag = nn.Linear(n_frag_feat, h_frag)
        self.embedding_fbond = nn.Linear(n_fbond_feat, h_fbond)
        self.mpnn = MPNN(h_frag + h_atom, h_fbond, n_layers)
        self.encode = nn.Linear(h_frag + h_atom, 2 * latent_dim)

    def forward(self, g_mol: GraphMol):
        g = g_mol.graph
        h_atoms = torch.cat([self.atom_encoder(g_atom) for g_atom in g_mol.graphs_atom], dim=0)

        h_frag = self.embedding_frag(g.ndata['n_feat'])
        h_frag = torch.cat([h_frag, h_atoms], dim=-1)

        h_fbond = self.embedding_fbond(g.edata['e_feat'])

        g.ndata['h'] = self.mpnn(g, h_frag, h_fbond)
        h_frag = dgl.mean_nodes(g, 'h')

        x = self.encode(h_frag).view(-1, 2, self.latent_dim)
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var


class MPNN(nn.Module):
    def __init__(self, h_node, h_edge, n_layers=2):
        super(MPNN, self).__init__()
        self.n_layer = n_layers
        edge_network = nn.Sequential(
            nn.Linear(h_edge, h_edge), nn.ReLU(),
            nn.Linear(h_edge, h_node * h_node)
        )
        self.conv = NNConv(
            h_node, h_node,
            edge_network, aggregator_type='mean', bias=False
        )
        self.gru = nn.GRU(h_node, h_node)

    def forward(self, g: dgl.DGLGraph, h_node: torch.Tensor, h_edge: torch.Tensor):
        h_gru = h_node.unsqueeze(0)
        for _ in range(self.n_layer):
            m = self.conv(g, h_node, h_edge)
            m = F.relu(m)
            h_node, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h_node = h_node.squeeze(0)
        return h_node


if __name__ == "__main__":
    from rdkit import Chem

    with open('test.txt', 'r') as f:
        smiles = [s.split()[0] for s in f.readlines()[:6]]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    dataset = [GraphMol(mol) for mol in mols]
    encoder = FragEncoder(n_atom_feat=17, n_bond_feat=5,
                          h_atom=128, h_bond=16,
                          n_frag_feat=2, n_fbond_feat=5,
                          h_frag=64, h_fbond=16)

    for g in dataset:
        print(g)
        z, mu, log_var = encoder(g)


