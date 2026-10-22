# 2. Redes neurais, CNNs e transfer learning

## O que é

### Rede neural em uma frase

Uma **rede neural** é uma função com milhões de parâmetros ajustáveis (os **pesos**). Ela recebe números (a imagem) e devolve números (ex.: máscaras). "Treinar" é ajustar esses pesos até a saída ficar boa.

### Como o treino funciona

1. **Forward**: passa uma imagem pela rede e obtém uma previsão.
2. **Loss** (função de perda): mede o quão errada a previsão está comparada ao gabarito (a anotação feita por humanos).
3. **Backpropagation**: calcula o **gradiente** — em que direção mexer cada peso para a loss diminuir.
4. **Otimizador** (ex.: SGD, Adam): dá um pequeno passo nessa direção. O tamanho do passo é a **learning rate**.
5. Repete para todas as imagens. Uma passada completa pelo dataset = **1 época** (epoch). As imagens são processadas em grupos chamados **batches**.

```
imagem ──▶ [ rede ] ──▶ previsão ──┐
                ▲                   ├─▶ loss ──▶ gradiente ──▶ ajusta pesos
                └───────────────────┘   gabarito
```

### Convolução e CNNs

Uma **convolução** desliza um pequeno filtro (ex.: 3×3) pela imagem, calculando em cada posição uma soma ponderada dos pixels vizinhos. Filtros diferentes respondem a padrões diferentes: bordas, cantos, texturas.

Uma **CNN** (Convolutional Neural Network) empilha muitas camadas de convolução:

- primeiras camadas → bordas e cores
- camadas do meio → texturas, formas simples (círculos, cantos)
- camadas profundas → partes de objetos ("isso parece uma agarra")

Modelos mais novos (como o encoder do SAM) usam **Transformers de visão (ViT)**, que dividem a imagem em pedaços (patches) e usam **atenção** para relacionar partes distantes da imagem. A ideia geral — camadas que extraem características cada vez mais abstratas — é a mesma.

### Backbone, neck e head

Detectores modernos são divididos em blocos:

- **Backbone**: extrai características (features) da imagem. É a parte "que enxerga".
- **Neck**: combina features de várias escalas (agarras pequenas e grandes).
- **Head**: transforma as features na saída final (caixas, classes, máscaras).

### Transfer learning e fine-tuning

Treinar uma rede do zero exige milhões de imagens. **Transfer learning** é reaproveitar uma rede já treinada em um dataset gigante (ex.: COCO, com 80 classes de objetos do dia a dia): o backbone dela já sabe reconhecer bordas, texturas e formas.

**Fine-tuning** é continuar o treino dessa rede com o *nosso* dataset (fotos de agarras), por poucas épocas. Com algumas centenas de imagens anotadas já dá para ter um bom detector.

## No nosso projeto

- **Fase 1 (SAM)**: não treinamos nada. Usamos os pesos prontos da Meta → isso se chama **inferência** (só o forward).
- **Fase 5 (YOLO)**: fazemos **fine-tuning** de um YOLO pré-treinado no COCO, usando um dataset de agarras.

## GPU e CUDA

Redes neurais fazem bilhões de multiplicações de matrizes — operações que uma **GPU** faz em paralelo muito mais rápido que a CPU. **CUDA** é a plataforma da NVIDIA que permite ao PyTorch usar a GPU.

Nossa máquina tem uma **RTX 4070 SUPER (12 GB de VRAM)** rodando em **WSL2**:

- No WSL2, o driver NVIDIA é o **do Windows**. **Não** instale driver NVIDIA dentro do Linux.
- Basta instalar o PyTorch com suporte a CUDA (comando em [pytorch.org/get-started](https://pytorch.org/get-started/locally/)).
- **VRAM** (memória da GPU) limita o tamanho do modelo e do batch. Se aparecer `CUDA out of memory`, reduza o batch, o tamanho da imagem ou use um modelo menor.

```python
import torch

print(torch.cuda.is_available())        # True = PyTorch enxerga a GPU
print(torch.cuda.get_device_name(0))    # "NVIDIA GeForce RTX 4070 SUPER"

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(3, 1024, 1024).to(device)  # tensor movido para a GPU
```

## Para ir além

- [3Blue1Brown — série "Neural Networks"](https://www.3blue1brown.com/topics/neural-networks) (a melhor intuição visual que existe)
- [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [CUDA no WSL — guia NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [CS231n — notas sobre CNNs](https://cs231n.github.io/convolutional-networks/)
