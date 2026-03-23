import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
import csv
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummary import summary
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix

# Preparando dirétorio
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

# Escolha do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Baixando dataset
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

# Renomeando os arquivos
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

# Variáveis estáticas
img_size = 224
batch_size_train = 32
batch_size_test = 1
num_epochs = 120
learning_rate = 1e-3
momentum_val = 0.9
num_classes = 120

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)

print(f"Número de classes: {len(train_dataset.classes)}")
print(f"Imagens de treino: {len(train_dataset)} | Lotes: {len(test_loader)}")
print(f"Imagens de teste: {len(test_dataset)} | Lotes: {len(train_loader)}")

class AlexNet(nn.Module):
  def __init__(self, num_classes=120):
    super(AlexNet, self).__init__()

    self.features = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
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
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

# Instanciando modelo
model = AlexNet(num_classes=num_classes).to(device)

# Sumário do modelo
summary(model=model, input_size=(3, img_size, img_size))

if __name__ == '__main__':
    print(f"Verificando diretórios...\nTreino: {train_dir}")
    

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_val)

    log_file = os.path.join(BASE_DIR, 'training_history.csv')
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc'])


df_training_history = pd.read_csv('training_history.csv')

plt.figure(figsize=(8,6))

plt.plot(df_training_history['epoch'], df_training_history['train_loss'], marker='o')
plt.title('Função de perda ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.plot(df_training_history['epoch'], df_training_history['train_acc'], marker='o')
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

#  Salvando modelo do AlexNet
torch.save(model.state_dict(), 'alexnet_dogs_final.pth')

y_true = []
y_pred = []

# Modelo em modo de avaliação
model.eval()

# Inferência do modelo
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Busca o maior tensor -> Saída é um logits
        _, predicted = torch.max(outputs, 1)

y_true.extend(labels.cpu().numpy())
y_pred.extend(predicted.cpu().numpy())

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

table_metrics = PrettyTable()
# Removi 'Accuracy' dos cabeçalhos da tabela
table_metrics.field_names = ['Classe', 'Precision', 'Recall', 'F1-Score', 'Support']

for chave, valores in report.items():
    # Ignora as métricas gerais para listar apenas as classes na tabela
    if chave not in ['accuracy', 'macro avg', 'weighted avg']:
        table_metrics.add_row([
            chave,
            # Removi a linha valores['accuracy'] que estava causando o erro
            round(valores['precision'], 3),
            round(valores['recall'], 3),
            round(valores['f1-score'], 3),
            round(valores['support'], 3),
        ])

print(table_metrics)

# A acurácia geral é acessada direto na raiz do relatório
print(f"\nAcurácia Global do Modelo: {round(report['accuracy'], 3)}")

cm = confusion_matrix(y_true, y_pred, normalize='true')

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title('Matriz de confusão')
plt.colorbar()

classes = np.unique(y_true)
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
            ha='center',
            va='center')
        
plt.tight_layout()
plt.show()

# Busca apenas 1 batch para amostras
images, labels = next(iter(test_loader))

indices = random.sample(range(len(images)), 3)

for idx in indices:

    image = images[idx].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        # Transforma logits em probabilidade
        probs = torch.softmax(outputs, dim=1)

        # Armazena os top 3
        top_probs, top_classes = torch.topk(probs, 3)

    table_top3 = PrettyTable()
    table_top3.field_names = ['Amostra aleatória', 'Classe real', 'Rank', 'Classe', 'Probabilidade']

    for i in range(3):
        table_top3.add_row([
            idx,
            labels[idx].item(),
            i+1,
            int(top_classes[0][i]),
            float(top_probs[0][i])
        ])

table_top3.title = "Top 3 probabilidades das amostras"
print(table_top3)