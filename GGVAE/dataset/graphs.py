import copy

import dgl
from rdkit import Chem
from collections import defaultdict
import torch
from GGVAE.common.chem import break_bond, Fragment

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


class GraphAtom(dgl.DGLGraph):
    def __init__(self, frag: Chem.RWMol):
        super(GraphAtom, self).__init__()

        # 添加节点
        num_atoms = frag.GetNumAtoms()
        atom_feats = zinc_nodes(frag)
        self.add_nodes(num=num_atoms, data=atom_feats)

        # 添加边
        for bond in frag.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            self.add_edges([u, v], [v, u])
        bond_feats = zinc_edges(frag, self.edges())
        self.edata.update(bond_feats)


def mol2frags(mol: Chem.RWMol):
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            continue

        try:
            skeleton, arm = break_bond(mol, u, v)
        except ValueError:
            continue


class GraphFrag(dgl.DGLGraph):
    def __init__(self, mol: Chem.RWMol):
        super(GraphFrag, self).__init__()
        self.mol = mol
        mol = Fragment(mol)



    def is_frag(self):
        pass
