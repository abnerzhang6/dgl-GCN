import dgl
import os
import random
import torch

from Tools import *
from torch.utils.data import random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.cuda.empty_cache()

g_list = []  # total dgl-graph
l_list = []  # total cors label

# 制作 label-tid 字典
with open(r'meta-dataset/twitter15/label.txt') as f:
    lines = f.readlines()
    label_dict = {}
    for line in lines:
        label, tid = line.split(':')
        label_dict[int(tid)] = label
    # 完成效果：label_dict[tid] -> label

emb_dim = 100  # embedding_dim
hid_dim = 200

filePath = r'meta-dataset/twitter15/tree'
fileList = os.listdir(filePath)

for fileName in fileList:

    with open(filePath + r'/' + fileName) as f:
        lines = f.readlines()
        tid = int(fileName[:-4])
        bereposters, reposters = [], []
        for line in lines:
            list1, list2 = line.split('->')
            if list1[1:-2].split(', ')[0][1:-1] != 'ROOT':
                bereposters.append(int(list1[1:-2].split(', ')[0][1:-1]))  # append send uid
                reposters.append(int(list2[1:-2].split(', ')[0][1:-1]))  # append receive uid
            # from a to b

        uid_dict = {}
        encoder = 1  # send-receive shared encoding system
        #  因为dgl.graph 只接受从0开始的node编码，故设置uid_dict

        for idx, con in enumerate(bereposters):
            if con in uid_dict.keys():
                bereposters[idx] = uid_dict[con]
            else:
                uid_dict[con] = encoder
                encoder += 1
                bereposters[idx] = uid_dict[con]

        for idx, con in enumerate(reposters):
            if con in uid_dict.keys():
                reposters[idx] = uid_dict[con]
            else:
                uid_dict[con] = encoder
                encoder += 1
                reposters[idx] = uid_dict[con]

        graph = dgl.graph((bereposters, reposters))
        
        graph.ndata['attr'] = torch.randn(graph.num_nodes(), emb_dim)
        # graph.ndata['label'] = torch.tensor([trans_label(label_dict[tid])] * graph.num_nodes())
        graph = dgl.add_self_loop(graph)
        g_list.append(graph.to(device))  # len = 1490
        l_list.append(torch.tensor(trans_label(label_dict[tid])).to(device))


# 制作 label-tid 字典
with open(r'meta-dataset/twitter16/label.txt') as f:
    lines = f.readlines()
    label_dict = {}
    for line in lines:
        label, tid = line.split(':')
        label_dict[int(tid)] = label
    # 完成效果：label_dict[tid] -> label

filePath = r'meta-dataset/twitter16/tree'
fileList = os.listdir(filePath)

for fileName in fileList:

    with open(filePath + r'/' + fileName) as f:
        lines = f.readlines()
        tid = int(fileName[:-4])
        bereposters, reposters = [], []
        for line in lines:
            list1, list2 = line.split('->')
            if list1[1:-2].split(', ')[0][1:-1] != 'ROOT':
                bereposters.append(int(list1[1:-2].split(', ')[0][1:-1]))  # append send uid
                reposters.append(int(list2[1:-2].split(', ')[0][1:-1]))  # append receive uid
            # from a to b

        uid_dict = {}
        encoder = 1  # send-receive shared encoding system
        #  因为dgl.graph 只接受从0开始的node编码，故设置uid_dict

        for idx, con in enumerate(bereposters):
            if con in uid_dict.keys():
                bereposters[idx] = uid_dict[con]
            else:
                uid_dict[con] = encoder
                encoder += 1
                bereposters[idx] = uid_dict[con]

        for idx, con in enumerate(reposters):
            if con in uid_dict.keys():
                reposters[idx] = uid_dict[con]
            else:
                uid_dict[con] = encoder
                encoder += 1
                reposters[idx] = uid_dict[con]

        graph = dgl.graph((bereposters, reposters))

        graph.ndata['attr'] = torch.randn(graph.num_nodes(), emb_dim)
        # graph.ndata['label'] = torch.tensor([trans_label(label_dict[tid])] * graph.num_nodes())
        graph = dgl.add_self_loop(graph)
        g_list.append(graph.to(device))  # len = 1490
        l_list.append(torch.tensor(trans_label(label_dict[tid])).to(device))

total_dataset = DglDataset(g_list, l_list)

generator = torch.Generator().manual_seed(42)
train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
len_dataset = len(total_dataset)
train_dataset, vali_dataset, test_dataset = random_split(total_dataset,
        lengths=[round(train_ratio * len_dataset), round(valid_ratio * len_dataset), round(test_ratio * len_dataset)],
        generator=generator)

BATCH_SIZE = 2048
train_dataloader = dgl.dataloading.GraphDataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
vali_dataloader = dgl.dataloading.GraphDataLoader(
    vali_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
test_dataloader = dgl.dataloading.GraphDataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

model = GCN(emb_dim, hid_dim, 4)
model.to(device)
opt = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
# ,eps = 1e-4
best_model_acc = -1

for epoch in range(10000):
    model.train()
    for idx, (batched_graph, batched_label) in enumerate(train_dataloader):
        feats = batched_graph.ndata['attr']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, batched_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if idx % 1 == 0:
            with torch.no_grad():
                vali_correct = 0
                model.eval()
                for bg, bl in vali_dataloader:
                    fts = bg.ndata['attr']
                    lgts = model(bg, fts)
                    pred = lgts.argmax(1)
                    vali_correct += sum(pred == bl)
                cur_acc = vali_correct / len(vali_dataset)
                print('epoch: {}, batch: {}, loss: {:.6f}, vali_acc: {:.6f}'.format(epoch + 1, idx + 1, loss.data, cur_acc))
        if best_model_acc < cur_acc:
            torch.save(model, 'model_save/model_best.pth')
    if epoch % 100 == 0:
        os.system('nvidia-smi')


with torch.no_grad():
    test_correct = 0
    model = torch.load('model_save/model_best.pth')
    model.eval()
    for bg, bl in test_dataloader:
        fts = bg.ndata['attr']
        lgts = model(bg, fts)
        pred = lgts.argmax(1)
        test_correct += sum(pred == bl)
        print('test_acc: {}'.format(test_correct / len(test_dataset)))
