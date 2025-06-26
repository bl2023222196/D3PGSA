import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List
from tqdm import tqdm
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import experience_generation_model
class Dataset(Dataset):
    def __init__(self, data_list, transform=None):

        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def data(data_list):
    dataset = Dataset(data_list)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    return dataloader

def time_gen(sigma, eps, N, T):
    return torch.tensor(
        [
            (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
            ** sigma
            for i in range(N)
        ]
    )

def train(data_list, arrival_rate, device = "cuda:0",n_epoch: int = 200,n_channels = 1):
    dataloader = data(data_list)
    model = Exp_Gen_Model(n_channels, D=256)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.00001)
    ema_model = experience_generation_model.Exp_Gen_Model(n_channels, D=256)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        time = time_gen(7.0, 0.002, N, 80.0).to(device)
        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            z = torch.randn_like(x)*10
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = time[t]
            t_1 = time[t + 1]
            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)
            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")
    torch.save(model.state_dict(), "./model/cm_model{}.pth".format(arrival_rate))
    torch.save(ema_model.state_dict(),"./model/cm_ema_model{}.pth".format(arrival_rate))
def generate(arrival_rate, device = "cuda:0",n_epoch: int = 2,n_channels = 1):
    model = ConsistencyModel(n_channels, D=256)
    model.load_state_dict(torch.load("./model/cm_model{}.pth".format(arrival_rate)))
    model.to(device)
    model.eval()
    E = []
    for _ in range(1, n_epoch):
        with torch.no_grad():
            xh = model.sample(
                torch.randn((32, 1, 24, 11)).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            E.append(xh)
    return E


