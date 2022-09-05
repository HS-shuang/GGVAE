import dgl
from rdkit import Chem
from collections import defaultdict
import torch
from typing import List

from GGVAE.my_dataset.chem import Fragment, get_frags

ATOM_TYPES = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

HYBRID_TYPES = [Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3]

BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC, None]


def zinc_nodes(mol) -> dict:
    atom_feats_dict = defaultdict(list)
    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        charge = atom.GetFormalCharge()
        symbol = atom.GetSymbol()
        atom_type = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)
        atom_feats_dict['node_charge'].append(charge)

        h_u = []
        h_u += [int(symbol == x) for x in ATOM_TYPES]
        h_u.append(atom_type)
        h_u.append(int(charge))
        h_u.append(int(aromatic))
        h_u += [int(hybridization == x) for x in HYBRID_TYPES]
        h_u.append(num_h)
        atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

    atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])
    atom_feats_dict['node_charge'] = torch.LongTensor(atom_feats_dict['node_charge'])
    return atom_feats_dict


def zinc_edges(mol, edges, self_loop=False) -> dict:
    bond_feats_dict = defaultdict(list)
    edges = [idxs.tolist() for idxs in edges]
    for e in range(len(edges[0])):
        u, v = edges[0][e], edges[1][e]
        if u == v and not self_loop:
            continue
        e_uv = mol.GetBondBetweenAtoms(u, v)
        bond_type = None if e_uv is None else e_uv.GetBondType()
        bond_feats_dict['e_feat'].append([float(bond_type == x) for x in BOND_TYPES])

    bond_feats_dict['e_feat'] = torch.FloatTensor(
        bond_feats_dict['e_feat'])
    return bond_feats_dict


def frag_nodes(frags: List[Fragment]):
    frag_feat_dic = defaultdict(list)
    for i, frag in enumerate(frags):
        frag_feat_dic['n_feat'].append(torch.FloatTensor([frag.GetNumAtoms(),
                                                          frag.GetRingInfo().NumRings()]))
    frag_feat_dic['n_feat'] = torch.stack(frag_feat_dic['n_feat'], dim=0)
    return frag_feat_dic


def frag_edges(mol, edges, self_loop=False) -> dict:
    bond_feats_dict = defaultdict(list)
    edges = [idxs.tolist() for idxs in edges]
    for e in range(len(edges[0])):
        u, v = edges[0][e], edges[1][e]
        if u == v and not self_loop:
            continue
        e_uv = mol.GetBondBetweenAtoms(u, v)
        bond_type = None if e_uv is None else e_uv.GetBondType()
        bond_feats_dict['e_feat'].append([1., 0., 0., 0., 0.])

    bond_feats_dict['e_feat'] = torch.FloatTensor(
        bond_feats_dict['e_feat'])
    return bond_feats_dict


def get_atoms_graph(frag: Chem.Mol) -> dgl.DGLGraph:
    g = dgl.graph([], idtype=torch.int32)
    # 添加节点
    num_atoms = frag.GetNumAtoms()
    g.add_nodes(num=num_atoms, data=zinc_nodes(frag))

    # 添加边
    for bond in frag.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        g.add_edges([u, v], [v, u])
    bond_feats = zinc_edges(frag, g.edges())
    g.edata.update(bond_feats)
    return g


def get_frags_graph(mol: Chem.Mol):
    g = dgl.graph([], idtype=torch.int32)
    frags = get_frags(Fragment(mol))

    # 添加节点
    g.add_nodes(num=len(frags), data=frag_nodes(frags))
    mapnum_fragidx = {num: idx for idx, frag in enumerate(frags) for num in frag.FragMapNums}


    # 添加边
    u, v = [], []
    for idx, frag in enumerate(frags):
        for bond in frag.FragBond:
            g.add_edges([idx, mapnum_fragidx[bond[1]]],
                        [mapnum_fragidx[bond[1]], idx])
            u.extend(bond)
            v.extend(bond[::-1])

    bond_feats = frag_edges(mol, g.edges())
    g.edata.update(bond_feats)
    return frags, g


class GraphMol:
    def __init__(self, mol: Chem.Mol):
        super(GraphMol, self).__init__()
        self.frags, self.graph= get_frags_graph(mol)
        self.graphs_atom = [get_atoms_graph(frag) for frag in self.frags]
        

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    import os
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1',
                   'O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1',
                   'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1',
                   'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1',
                   'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl',
                   'CCOCCCNC(=O)c1cc(OC)ccc1Br',
                   'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1',
                   'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F',
                   'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N',
                   'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1',
                   'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1',
                   'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1',
                   'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1',
                   'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']
    mols = [Chem.MolFromSmiles(smile) for smile in test_smiles]
    gs = [GraphMol(mol) for mol in mols]
    for i in range(len(test_smiles)):
        print(gs[i].graphs_atom.edata['e_feat'])
        print(gs[i])
        plt.figure()
        nx.draw(gs[i].to_networkx(), with_labels=True)
        if not os.path.exists('plt'):
            os.makedirs('plt')
        plt.savefig(f'plt/nx{i}.jpg')
        plt.show()

