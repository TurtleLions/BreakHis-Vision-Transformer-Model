# TRAINING AND TESTING

import os
import numpy as np
import pandas as pd
from PIL import Image
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import OneCycleLR

# dataset
class BreakHisDataset(Dataset):
  def __init__(self, csv_file, root_dir, train=True, transform=None):
    
    self.data_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

    if train:
      self.data_frame = self.data_frame[self.data_frame['grp'].str.lower() == "train"]
    else:
      self.data_frame = self.data_frame[self.data_frame['grp'].str.lower() == "test"]
    
    self.data_frame.reset_index(drop=True, inplace=True)

  def __len__(self):
    return len(self.data_frame)

  def __getitem__(self, idx):
    row = self.data_frame.iloc[idx]
    filename = row['filename']
    
    img_path = os.path.join(self.root_dir, filename)
    image = Image.open(img_path).convert('RGB')
  
    lower_filename = filename.lower()
    if "adenosis" in lower_filename:
      label = 0
    elif "fibroadenoma" in lower_filename:
      label = 1
    elif "phyllodes_tumor" in lower_filename:
      label = 2
    elif "tubular_adenoma" in lower_filename:
      label = 3
    elif "ductal_carcinoma" in lower_filename:
      label = 4
    elif "lobular_carcinoma" in lower_filename:
      label = 5
    elif "mucinous_carcinoma" in lower_filename:
      label = 6
    elif "papillary_carcinoma" in lower_filename:
      label = 7
    else:
      raise ValueError(f"Cannot determine label from filename: {filename}")

    if self.transform:
      image = self.transform(image)
    return image, label

# ViT components
class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()
    self.d_model = d_model
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.linear_project = nn.Conv2d(self.n_channels, self.d_model,
                                    kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x):
    x = self.linear_project(x)
    x = x.flatten(2)
    x = x.transpose(1, 2)
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    pe = torch.zeros(max_seq_length, d_model)
    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos / (10000 ** (i / d_model)))
        else:
          pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    tokens_batch = self.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat((tokens_batch, x), dim=1)
    x = x + self.pe
    return x

class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size
    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
    attention = Q @ K.transpose(-2, -1)
    attention = attention / (self.head_size ** 0.5)
    attention = torch.softmax(attention, dim=-1)
    out = attention @ V
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads
    self.W_o = nn.Linear(d_model, d_model)
    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.W_o(out)
    return out

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4, dropout_prob=0.1):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(d_model, n_heads)
    self.dropout1 = nn.Dropout(dropout_prob)
    self.ln2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model * r_mlp),
        nn.GELU(),
        nn.Dropout(dropout_prob),
        nn.Linear(d_model * r_mlp, d_model),
        nn.Dropout(dropout_prob)
    )

  def forward(self, x):
    attn_out = self.mha(self.ln1(x))
    x = x + self.dropout1(attn_out)
    mlp_out = self.mlp(self.ln2(x))
    x = x + mlp_out
    return x

class VisionTransformer(nn.Module):
  def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, dropout_prob=0.1):
    super().__init__()
    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model
    self.n_classes = n_classes
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_channels = n_channels
    self.n_heads = n_heads
    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1

    self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
    self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
    self.transformer_encoder = nn.Sequential(
        *[TransformerEncoder(self.d_model, self.n_heads, r_mlp=4, dropout_prob=dropout_prob)
          for _ in range(n_layers)]
    )
    self.classifier = nn.Linear(self.d_model, self.n_classes)

  def forward(self, images):
    x = self.patch_embedding(images)
    x = self.positional_encoding(x)
    x = self.transformer_encoder(x)
    x = self.classifier(x[:, 0])
    return x

if __name__ == '__main__':
  import torch.multiprocessing as mp
  mp.freeze_support()
      
  # hyperparameters
  d_model = 36
  n_classes = 8  # e.g., nonbinary classification: 0 for adenosis, 1 for fibroadenoma, 2 for phyllodes_tumor, 3 for tubular_adenoma, 4 for ductal_carcinoma, 5 for lobular_carcinoma, 6 for mucinous_carcinoma, 7 for papillary_carcinoma
  img_size = (256, 256)
  patch_size = (16, 16)
  n_channels = 3
  n_heads = 12
  n_layers = 12
  batch_size = 128
  epochs = 300
  alpha = 0.001
  dropout_prob = 0.1

  train_transform = T.Compose([
      T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
      T.RandomHorizontalFlip(),
      T.RandomVerticalFlip(),
      T.RandomRotation(degrees=15),
      T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      T.ToTensor(),
      T.Normalize(mean=[0.7872, 0.6222, 0.7640], std=[0.1005, 0.1330, 0.0837])
  ])


  test_transform = T.Compose([
      T.Resize(img_size),
      T.ToTensor(),
      T.Normalize(mean=[0.7872, 0.6222, 0.7640], std=[0.1005, 0.1330, 0.0837])
  ])


  csv_file = "Folds.csv"
  root_dir = "./../BreaKHis_v1/"

  train_set = BreakHisDataset(csv_file=csv_file, root_dir=root_dir, train=True, transform=train_transform)
  train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
  cudnn.benchmark = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.is_available():
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})")
  else:
    print("Using device:", device)

  transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, dropout_prob).to(device)
  optimizer = AdamW(transformer.parameters(), lr=alpha, weight_decay=1e-4)
  scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)
  criterion = nn.CrossEntropyLoss()

  total_start = time.time()

  # Train loop
  for epoch in range(epochs):
    start = time.time()
    transformer.train()
    training_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = transformer(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      scheduler.step()
      training_loss += loss.item()
    end = time.time()
    curr_lr = scheduler.get_last_lr()
    print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss / len(train_loader):.3f} time: {end-start:.2f} sec lr: {curr_lr}')
    if ((epoch+1) % 10) == 0:
      model_scripted = torch.jit.script(transformer)
      model_scripted.save(f"./checkpoints/checkpoint:{epoch+1}-{d_model}-{n_classes}-{img_size}-{patch_size}-{n_channels}-{n_heads}-{n_layers}-{batch_size}-{epochs}-{alpha}-{dropout_prob}.pth")

  # Testing

  test_set = BreakHisDataset(csv_file=csv_file, root_dir=root_dir, train=False, transform=test_transform)
  test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)


  def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nModel Accuracy: {accuracy:.2f}%')
    return accuracy

  test_accuracy = test_model(transformer, test_loader, device)

  # Save the final model
  model_scripted = torch.jit.script(transformer)
  model_scripted.save(f"./models/nonbinary/{test_accuracy:.2f}%-{d_model}-{n_classes}-{img_size}-{patch_size}-{n_channels}-{n_heads}-{n_layers}-{batch_size}-{epochs}-{alpha}-{dropout_prob}.pth")

  total_end = time.time()

  print(f'time: {total_end-total_start:.2f}')