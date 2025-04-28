import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from collections import deque


def data_augmentation(x, noise_level=0.5, mask_prob=0.1):
    """
    对输入数据进行数据增强。

    参数:
    - x: 输入数据，形状为 (batch, 1, 11)
    - noise_level: 添加到数据的噪声水平
    - mask_prob: 每个特征被遮挡的概率

    返回:
    - 增强后的数据
    """
    # 添加随机噪声
    noise = torch.randn_like(x) * noise_level
    x_noised = x + noise

    # 随机遮挡
    mask = torch.rand_like(x) < mask_prob
    x_masked = x_noised.masked_fill(mask, 0)

    return x_masked
class CustomDataset(Dataset):
    def __init__(self, data_deque):
        self.data = list(data_deque)  # 将deque转换为列表以方便索引

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回对应的数据项
        # 根据实际需要，这里可以添加任何必要的预处理步骤
        return self.data[idx]

class MLP(nn.Module):
    def __init__(self, input_dim=11, feature_dim=256):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x):
        return self.layers(x)


class SimSiam(nn.Module):
    def __init__(self, input_dim=11, feature_dim=256, pred_dim=256):
        super(SimSiam, self).__init__()
        self.encoder = MLP(input_dim=input_dim, feature_dim=feature_dim)

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, feature_dim)
        )

    def forward(self, x):
        x1 = data_augmentation(x)
        x2 = data_augmentation(x)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
def sim_siam_loss(p1, p2, z1, z2):
    criterion = nn.CosineSimilarity(dim=1)
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    return loss

# 假设 dataloader 是你的数据加载器
def train(my_deque, arrival_rate):
    loss_list=[]
    model = SimSiam(input_dim=11).to(device)
    # model.load_state_dict(torch.load("./model/simsiam.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    custom_dataset = CustomDataset(my_deque)

    # 创建DataLoader
    dataloader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)
    for epoch in range(100):
        for batch in dataloader:
            x = batch # 假设batch直接是数据，没有标签

            optimizer.zero_grad()
            p1, p2, z1, z2 = model(x)
            loss = sim_siam_loss(p1, p2, z1, z2)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss_list.append(loss.item())
    torch.save(model.state_dict(), './model/simsiam{}.pth'.format(arrival_rate))
    return loss_list
def test(arrival_rate):
    model = SimSiam(input_dim=11).to(device)
    model.load_state_dict(torch.load("./model/simsiam{}.pth".format(arrival_rate)))
    return model
