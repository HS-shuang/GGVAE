from torch.utils import data
import dgl


class GraphDataset(data.Dataset):
    def __init__(self, graphs):
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    @staticmethod
    def collate_fn(batch):
        g = dgl.batch(batch)
        return g
