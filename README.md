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
