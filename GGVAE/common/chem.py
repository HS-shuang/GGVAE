import copy
from rdkit import Chem
from collections import deque


MAX_SIZE = 10


# validity
def standardize_smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def check_validity(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    if not isinstance(mol, Chem.Mol): return False
    if mol.GetNumBonds() < 1: return False
    try:
        # Chem.SanitizeMol(mol,
        #     sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.SanitizeMol(mol)
        Chem.RemoveHs(mol)
        return True
    except ValueError:
        return False


class Fragment(Chem.RWMol):
    def __init__(self, mol, outer=set()):
        super(Fragment, self).__init__(mol)
        self.outer = outer

    def add_outer(self, outer):
        self.outer.add(outer)


def break_bond(mol, u, v):
    mol = Chem.RWMol(copy.deepcopy(mol))
    bond = mol.GetBondBetweenAtoms(u, v)
    bond_type = bond.GetBondType()
    if not bond_type == Chem.rdchem.BondType.SINGLE:
        raise ValueError
    mol.RemoveBond(u, v)

    mapping = []
    frags = list(Chem.rdmolops.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=mapping))
    mapping = [list(m) for m in mapping]
    if not len(frags) == 2:
        raise ValueError
    if u not in mapping[0]:
        mapping = [mapping[1], mapping[0]]
        frags = [frags[1], frags[0]]

    # re-index
    u = mapping[0].index(u) 
    v = mapping[1].index(v)
    
    # standardizing frags will cause wrong indexing for u and v
    f1 = Fragment(frags[0], {u})
    f2 = Fragment(frags[1], {v})
    return f1, f2


def get_frags(mol: Fragment):
    dfs = deque([mol])
    frags = []
    while dfs:
        mol_now = dfs.popleft()
        for bond in mol_now.GetBonds():
            if bond.IsInRing() or not bond.GetBondType() == Chem.BondType.SINGLE:
                continue

            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a1.IsInRing() or a2.IsInRing():
                try:
                    frag1, frag2 = break_bond(mol_now, a1.GetIdx(), a2.GetIdx())
                    dfs.append(frag1)
                    dfs.append(frag2)
                    break
                except ValueError:
                    continue
        else:
            frags.append(mol_now)
    return frags


def combine(skeleton, arm):
    mol = Chem.CombineMols(skeleton, arm)
    mol = Chem.RWMol(mol)
    u = skeleton.outer.pop()
    v = skeleton.GetNumAtoms() + arm.outer.pop()
    mol.AddBond(u, v, Chem.rdchem.BondType.SINGLE)
    return mol.GetMol()


if __name__ == '__main__':
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles('CCCCn1c(=O)[nH]c2cc(C(=O)NCC3OC(n4cnc5c(NCc6ccc(Oc7ccccc7)cc6)ncnc54)C(O)C3O)ccc21')
    # mol = Chem.MolFromSmiles('COc1ccc(Cn2c(CCc3ccccc3)nnc2C(Cc2c[nH]c3ccccc23)NC(=O)c2cc3ccccc3cn2)cc1')

    frags = get_frags(Fragment(mol))
    print(mol.GetNumAtoms())
    print(sum(m.GetNumAtoms() for m in frags))
    Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(500, 150)).save('0.jpg')
    Draw.MolsToGridImage(frags, subImgSize=(200, 150)).save('1.jpg')

