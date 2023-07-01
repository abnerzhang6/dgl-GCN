import dgl
# import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import SumPooling
import torch.nn as nn
import torch.nn.functional as F


class DglDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv5 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv6 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv7 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv8 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv9 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv10 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv11 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv12 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv2(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv3(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv4(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv5(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv6(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv7(g, h))
        # h = self.dropout(h)
        h = F.relu(self.conv8(g, h))
        # h = F.relu(self.conv9(g, h))
        # h = F.relu(self.conv10(g, h))
        # h = F.relu(self.conv11(g, h))
        # h = F.relu(self.conv12(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.readout_nodes(g, 'h')
            return self.classify(hg)


def trans_label(label):
    d = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}
    return d[label]


# def show_graph(g, title, label):
#     fig, ax = plt.subplots()
#     nx.draw(g.to_networkx(), ax=ax)
#     ax.set_title(title)
#     if label == "false":
#         fig.savefig(r"D:\pycharmProjects\pythonProject\dgl-learn\tu\false" + "\\" + title + ".png")
#     elif label == "non-rumor":
#         fig.savefig(r"D:\pycharmProjects\pythonProject\dgl-learn\tu\non-rumor" + "\\" + title + ".png")
#     elif label == "true":
#         fig.savefig(r"D:\pycharmProjects\pythonProject\dgl-learn\tu\true" + "\\" + title + ".png")
#     elif label == "unverified":
#         fig.savefig(r"D:\pycharmProjects\pythonProject\dgl-learn\tu\unverified" + "\\" + title + ".png")
