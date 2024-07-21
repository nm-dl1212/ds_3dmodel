# パッケージの読み込み
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.utils import scatter
from torch_geometric.typing import WITH_TORCH_CLUSTER


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch,
                          batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        # x_dst = None if x is None else x[idx]
        # x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        # pos, batch = pos[idx], batch[idx]

        x = self.conv(x, pos, edge_index)

        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 16, 16, 32]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([32 + 3, 32, 32, 64]))
        self.sa3_module = GlobalSAModule(MLP([64 + 3, 64, 128, 256]))
        self.mlp = MLP([256, 128, 64, 2], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)


def train(model, loader, optimizer):
    model.train()
    for data in loader:
        # data = data.to(DEVICE)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, data.category)
        loss.backward()
        optimizer.step()

    return loss


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        # data = data.to(DEVICE)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.category).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # データのダウンロード
    DATA_REPO = 'ShapeNet'  # ['ModelNet', 'ShapeNet']
    if DATA_REPO == 'ModelNet':
        path = 'data/ModelNet10'
        pre_transform = T.NormalizeScale()
        transform = T.Compose([T.SamplePoints(512), T.ToDevice(DEVICE)])
        train_dataset = ModelNet(path, '10', True, transform, pre_transform)
        test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    elif DATA_REPO == 'ShapeNet':
        path = 'data/ShapeNet'
        categories = ["Airplane", "Car"]
        pre_transform = T.NormalizeScale()
        transform = T.ToDevice(DEVICE)
        train_dataset = ShapeNet(path, categories, split='train', include_normals=False,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ShapeNet(path, categories, split='test', include_normals=False,
                                pre_transform=pre_transform, transform=transform)

    # loader
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=6)

    model = Net().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train_loss = train(model, train_loader, optimizer)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')
