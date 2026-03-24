# AlexNet - Stanford Dogs Dataset

## Descrição do Projeto

Este projeto implementa e avalia uma rede neural convolucional baseada na arquitetura AlexNet para classificação do Stanford Dogs Dataset, que contém imagens de 120 raças de cães. O objetivo é comparar o desempenho do modelo com e sem a camada softmax explícita na saída, analisando métricas de treinamento e teste.

## Principais Conclusões

- **Overfitting severo:**
  - O modelo AlexNet atingiu cerca de 98% de acurácia no conjunto de treinamento, mas apenas 0,6% no conjunto de teste.
  - As predições no teste ficaram concentradas em poucas classes, com precisão, recall e F1-score próximos de zero para a maioria das classes.
  - Isso evidencia overfitting, causado pela alta capacidade do modelo (cerca de 60 milhões de parâmetros) e pelo tamanho limitado do dataset (~12 mil imagens).

- **Softmax explícito vs. implícito:**
  - O modelo original do AlexNet utiliza uma camada softmax na saída, mas a função de perda CrossEntropyLoss já incorpora o cálculo do log-softmax.
  - Testes mostraram que adicionar softmax explícito prejudica o início do treinamento (loss inicial muito alta), mas ambos os modelos convergem para acurácia próxima de 100% no treino.
  - No entanto, ambos apresentam overfitting severo no teste.

- **Regularização:**
  - A adição de weight decay (L2) ao otimizador não foi suficiente para evitar o overfitting.

## Recomendações para Melhorar o Modelo

1. **Data Augmentation:**
  - Aplicar transformações nas imagens (rotação, flip, zoom, etc.) para aumentar a variabilidade dos dados e ajudar na generalização.
2. **Transfer Learning:**
  - Utilizar modelos pré-treinados em grandes datasets (como ImageNet) para aproveitar conhecimento prévio e melhorar o desempenho.
3. **Batch Normalization:**
  - Inserir camadas de batch normalization para estabilizar o treinamento e reduzir o overfitting.

## Visualizações

- Gráficos de loss e acurácia ao longo das épocas mostram que ambos os modelos aprendem no treino, mas não generalizam para o teste.
- O modelo com softmax explícito apresenta loss inicial mais alta e aprendizado mais lento no início, mas ambos convergem.

## Como Executar

1. Instale as dependências:
  ```sh
  pip install -r requirements.txt
  ```
2. Execute o notebook `atv-topicos1.ipynb` para treinar e avaliar o modelo.
3. Use o script `graf.py` para visualizar os gráficos de loss e acurácia dos dois experimentos.

## Estrutura dos Arquivos

- `atv-topicos1.ipynb`: Notebook principal com todo o pipeline de treinamento, avaliação e análise.
- `csv/training_history.csv`: Histórico de treino sem softmax explícito.
- `csv/training_history_1teste-overfit.csv`: Histórico de treino com softmax explícito.
- `graf.py`: Script para plotar os gráficos comparativos.

## Equipe
- Adriana Raffaella
- Cristiano Peniche
- Henrique Furtado
- Paulo Andrade

---

*Projeto desenvolvido para a disciplina Tópicos para Computação 1 - 2026.1, Escola Superior de Tecnologia, Profa. Dra. Elloá B. Guedes.*
# Classificação de Raças de Cachorros com CNN (Stanford Dogs Dataset)

Projeto desenvolvido para a disciplina **Tópicos para Computação 1 – 2026.1** da **Escola Superior de Tecnologia**. O objetivo é implementar e treinar uma **Rede Neural Convolucional (CNN)** para classificação de imagens de diferentes raças de cachorros utilizando o **Stanford Dogs Dataset**.

## Informações da Atividade

- **Disciplina:** Tópicos para Computação 1  
- **Instituição:** Escola Superior de Tecnologia  
- **Professora:** Profa. Dra. Elloá B. Guedes  
- **Data da atividade:** 19/03/2026  
- **Data de entrega:** 23/03/2026  

## Equipe

- Adriana Raffaella  
- Cristiano Peniche  
- Henrique Furtado  
- Paulo Andrade  

---

# Objetivo do Projeto

Construir e treinar uma **rede neural convolucional (CNN)** capaz de classificar imagens em **120 diferentes raças de cachorros** utilizando o **Stanford Dogs Dataset**, que possui mais de **20 mil imagens** para treino e teste.

O projeto inclui:

- Download automático do dataset  
- Pré-processamento das imagens  
- Criação dos DataLoaders  
- Treinamento do modelo  
- Avaliação com métricas de classificação  

---

# Dataset

O dataset utilizado é uma versão organizada do **Stanford Dogs Dataset** disponível no Kaggle.

Dataset original:  
http://vision.stanford.edu/aditya86/ImageNetDogs/

Versão utilizada no projeto:  
https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-traintest

Características:

- **120 classes (raças de cães)**  
- **Mais de 20.000 imagens**  
- Separação em:
  - `train`
  - `test`

---

# Como Executar o Projeto

1. Criar e ativar ambiente virtual

```bash
python -m venv .venv
.venv/Scripts/Activate #No windows
```

2. Instalar dependências

```bash
pip install -r requirements.txt
```

3. Abrir o notebook e executar as células

## Referências

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Stanford Dogs Dataset no Kaggle](https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-traintest)
