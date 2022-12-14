import os
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem

from .utils import load_mols
from GGVAE.my_dataset.chem import get_frags, Fragment


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default='GGVAE/my_dataset')
    parser.add_argument('--mols_file',  type=str,   default='chembl_.txt')
    parser.add_argument('--vocab_name', type=str,   default='chembl_',)
    parser.add_argument('--max_size',   type=int,   default=10, help='max size of arm')
    parser.add_argument('--test', type=bool, default=True, help='是否保存词汇')
    args = parser.parse_args()

    # load data
    mols = load_mols(args.data_dir, args.mols_file)
    vocab = {}  # dict{smiles: [Frage, cont]}

    for mol in tqdm(mols):
        frags = get_frags(Fragment(mol))
        smiles = [Chem.MolToSmiles(frag) for frag in frags]
        for i, s in enumerate(smiles):
            if s in vocab:
                vocab[s][1] += 1
            else:
                vocab[s] = [frags[i], 1]

    if not args.test:
        print('save...')

        # save frags
        smiles = [s for s in vocab]
        smiles.sort(key=lambda x: vocab[x][1], reverse=True)
        frags = [vocab[s][0] for s in smiles]
        vocab_dir = os.path.join(args.data_dir, 'vocab_%s' % args.vocab_name)
        os.makedirs(vocab_dir, exist_ok=True)

        with open(os.path.join(vocab_dir, 'frags.pkl'), 'wb') as f:  # list of Arm
            pickle.dump(frags, f)
        with open(os.path.join(vocab_dir, 'frags.smi'), 'w') as f:
            for s in smiles:
                f.write('%i\t%s\n' % (vocab[s][1], s))
    print('finished!')
