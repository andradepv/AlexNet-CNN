
import pandas as pd
import matplotlib.pyplot as plt

# Caminhos dos arquivos CSV

# Identificação dos modelos
csv1 = 'csv/training_history.csv'  # Sem mexer no softmax (softmax removido)
csv2 = 'csv/training_history_1teste-overfit.csv'  # Com softmax explícito
label1 = 'Sem Softmax explícito'
label2 = 'Com Softmax explícito'

# Lê os dados
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# 1. Loss vs Épocas
plt.figure(figsize=(8,5))
plt.plot(df1['epoch'], df1['train_loss'], label=label1)
plt.plot(df2['epoch'], df2['train_loss'], label=label2)
plt.title('Loss ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Acurácia vs Épocas
plt.figure(figsize=(8,5))
plt.plot(df1['epoch'], df1['train_acc'], label=label1)
plt.plot(df2['epoch'], df2['train_acc'], label=label2)
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Loss + Acurácia juntos (para cada modelo)
fig, ax1 = plt.subplots(figsize=(8,5))
color1 = 'tab:blue'
color2 = 'tab:orange'
ax1.set_xlabel('Épocas')
ax1.set_ylabel(f'Loss ({label1})', color=color1)
ax1.plot(df1['epoch'], df1['train_loss'], color=color1, label=f'Loss {label1}')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel(f'Acurácia ({label1})', color=color2)
ax2.plot(df1['epoch'], df1['train_acc'], color=color2, label=f'Acurácia {label1}')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f'Treinamento: {label1}')
fig.tight_layout()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(8,5))
color1 = 'tab:green'
color2 = 'tab:red'
ax1.set_xlabel('Épocas')
ax1.set_ylabel(f'Loss ({label2})', color=color1)
ax1.plot(df2['epoch'], df2['train_loss'], color=color1, label=f'Loss {label2}')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel(f'Acurácia ({label2})', color=color2)
ax2.plot(df2['epoch'], df2['train_acc'], color=color2, label=f'Acurácia {label2}')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f'Treinamento: {label2}')
fig.tight_layout()
plt.grid(True)
plt.show()
