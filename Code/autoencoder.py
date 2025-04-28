import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class CustomDataset(Dataset):
    def __init__(self, data_deque):
        self.data = list(data_deque)  # 将deque转换为列表以方便索引

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回对应的数据项
        # 根据实际需要，这里可以添加任何必要的预处理步骤
        return self.data[idx]
class Autoencoder(nn.Module):
    def __init__(self, input_dim, feature_dim=256, output_dim=256):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(True)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(256, 516),
            nn.ReLU(True),
            nn.Linear(516, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入到 (batch_size, 11)
        z1 = self.encoder(x)  # 编码到隐空间
        z2 = self.encoder(x)
        x_reconstructed = self.decoder(z1)  # 解码回原始空间
        x_reconstructed = x_reconstructed.view(x.size(0), 1, 11)  # 重新调整形状
        return z1, x_reconstructed


def train(my_deque, arrival_rate, num_epochs=100):
    model = Autoencoder(input_dim=11).to(device)
    # model.load_state_dict(torch.load("./model/simsiam.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    custom_dataset = CustomDataset(my_deque)

    # 创建DataLoader
    dataloader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            input_data = data[0]
            optimizer.zero_grad()
            z, output = model(input_data)
            criterion = nn.MSELoss()
            loss = criterion(output, input_data)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'第 {epoch + 1}/{num_epochs} 轮，损失: {avg_loss:.4f}')
    torch.save(model.state_dict(), './model/autoencoder{}.pth'.format(arrival_rate))

def test(arrival_rate):
    model = Autoencoder(input_dim=11).to(device)
    model.load_state_dict(torch.load("./model/autoencoder{}.pth".format(arrival_rate)))
    return model







