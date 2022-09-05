import copy
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
    def __init__(self, mol: Chem.Mol, Bond=set()):
        super(Fragment, self).__init__(mol)
        self.FragBond = Bond  # set of tuple，（连接处位于自己的原子map num, 连接处位于父fragment的原子map num）
        self.FragMapNums = None

    def updateMapNum(self):
        self.FragMapNums = {atom.GetAtomMapNum() for atom in self.GetAtoms()}


def break_bond(mol: Fragment, u, v):
    root_bond = mol.FragBond
    mol = Chem.RWMol(copy.deepcopy(mol))
    bond = mol.GetBondBetweenAtoms(u, v)
    bond_type = bond.GetBondType()
    if not bond_type == Chem.rdchem.BondType.SINGLE:
        raise ValueError
    mol.RemoveBond(u, v)

    # 确定 u, v 的 map num
    u, v = mol.GetAtomWithIdx(u).GetAtomMapNum(), mol.GetAtomWithIdx(v).GetAtomMapNum()

    frags = list(Chem.rdmolops.GetMolFrags(mol, asMols=True))
    if not len(frags) == 2:
        raise ValueError

    mapping = [{atom.GetAtomMapNum() for atom in frag.GetAtoms()} for frag in frags]

    if u not in mapping[0]:  # 确保原子u在 f1 中
        frags = [frags[1], frags[0]]

    f1_bond = set(bond for bond in root_bond if bond[0] in mapping[0])
    f2_bond = set(bond for bond in root_bond if bond[0] in mapping[1])
    f2_bond.add((v, u))
    # standardizing frags will cause wrong indexing for u and v
    f1 = Fragment(frags[0], Bond=f1_bond)
    f2 = Fragment(frags[1], Bond=f2_bond)
    return f1, f2


def get_frags(mol: Chem.Mol):
    mol = Fragment(mol)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    stack = [mol]
    frags = []
    while stack:
        mol_now = stack.pop()
        for bond in mol_now.GetBonds():
            if bond.IsInRing() or not bond.GetBondType() == Chem.BondType.SINGLE:
                continue

            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a1.IsInRing() or a2.IsInRing():
                try:
                    frag1, frag2 = break_bond(mol_now, a1.GetIdx(), a2.GetIdx())
                    if frag1.GetNumAtoms() > 1 and frag2.GetNumAtoms() > 1:
                        stack.append(frag2)
                        stack.append(frag1)
                    else:
                        continue
                    break
                except ValueError:
                    continue
        else:
            mol_now.updateMapNum()
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
    mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    #mol = Chem.MolFromSmiles('O=C1c2cc3c(cc2-c2c1c1ccccc1c(=O)n2CCCOC(=O)C(F)(F)F)OCO3')

    frags_list = [get_frags(mol) for mol in mols]

    for i, mol in enumerate(mols):
        frags = frags_list[i]
        print(i, mol.GetNumAtoms(), sum(m.GetNumAtoms() for m in frags))
        Draw.MolsToGridImage([mol], molsPerRow=1,
                             subImgSize=(500, 150)).save(f'jpg/{i}-00.jpg')
        Draw.MolsToGridImage([Chem.RWMol(frag) for frag in frags],
                             subImgSize=(200, 150),).save(f'jpg/{i}-11.jpg')

