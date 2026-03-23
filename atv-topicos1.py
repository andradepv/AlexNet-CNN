import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import kagglehub
import os
import shutil
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base Directory: {BASE_DIR}")
local_path = os.path.join(BASE_DIR, "dataset_dogs")
print(f"Local Path: {local_path}")
base_path = os.path.join(local_path, "cropped")
print(f"Base Path: {base_path}")    
train_dir = os.path.join(base_path, "train")
print(f"Train Directory: {train_dir}")
test_dir = os.path.join(base_path, "test")
print(f"Test Directory: {test_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(local_path):
    print("Baixando dataset...")
    try:
        downloaded_path = kagglehub.dataset_download("miljan/stanford-dogs-dataset-traintest")
        print(f"Copiando arquivos para: {local_path}")
        shutil.copytree(downloaded_path, local_path)
    except Exception as e:
        print(f"Erro no download/cópia: {e}")

pasta_redundante = os.path.join(base_path, "cropped")
if os.path.exists(pasta_redundante):
    shutil.rmtree(pasta_redundante)

if os.path.exists(base_path):
    for subpasta in ['train', 'test']:
        path_alvo = os.path.join(base_path, subpasta)
        if os.path.exists(path_alvo):
            for folder_name in os.listdir(path_alvo):
                old_path = os.path.join(path_alvo, folder_name)
                if os.path.isdir(old_path) and '-' in folder_name:
                    new_name = folder_name.split('-', 1)[1]
                    new_path = os.path.join(path_alvo, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)

img_size = 224
batch_size_train = 32
batch_size_test = 1
num_epochs = 120
learning_rate = 1e-3
momentum_val = 0.9
num_classes = 120

data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                         std=[0.5, 0.5, 0.5])
])

class AlexNet(nn.Module):
    def __init__(self, num_classes=120):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    print(f"Verificando diretórios...\nTreino: {train_dir}")
    

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size_test, 
        shuffle=False, 
        num_workers=0
    )

    print(f"✅ ImageHandlers criados com sucesso!")
    print(f"📊 Imagens de Treino: {len(train_dataset)} | Lotes: {len(train_loader)}")
    print(f"📊 Imagens de Teste: {len(test_dataset)} | Lotes: {len(test_loader)}")

    print(f"Classes encontradas: {len(train_dataset.classes)}")
    
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_val)

    log_file = os.path.join(BASE_DIR, 'training_history.csv')
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc'])


torch.save(model.state_dict(), 'alexnet_dogs_final.pth')
print("\n💾 Modelo salvo como 'alexnet_dogs_final.pth'")
