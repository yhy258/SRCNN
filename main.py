import torch
from data_load import SR_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SRCNN
from tqdm.notebook import tqdm
from config import Config
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

hr_path = Config.hr_path
lr_path = Config.lr_path
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = SR_dataset(
  lr_path = lr_path,
  hr_path = hr_path,
  transform = transform,
  interpolation_mode=Config.interpolation_mode,
  interpolation_scale=Config.interpolation_scale
)

train_loader = DataLoader(
  train_dataset,
  batch_size = Config.batch_size,
  shuffle =True
)

model = SRCNN().to(DEVICE)
optimizer =torch.optim.Adam(model.parameters(), lr = Config.lr)


epochs = Config.epochs
model.train()
for epoch in range(epochs):
  print("{}/{} EPOCHS".format(epoch+1, epochs))
  for x,y in tqdm(train_loader):
    x = x.to(DEVICE)

    y = y.to(DEVICE)
    pred = model(x)

    loss = torch.nn.functional.mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(loss.item())