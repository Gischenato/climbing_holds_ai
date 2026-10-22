# 1. Visão computacional: imagens e tarefas

## O que é

**Visão computacional** é a área que ensina computadores a extrair informação de imagens. Para isso, primeiro precisamos entender como o computador "vê" uma imagem.

### Imagem = matriz de números

Uma foto é uma grade de **pixels**. Cada pixel colorido tem 3 números (canais) — **R**ed, **G**reen, **B**lue — normalmente de 0 a 255.

```
Foto 4000 × 3000 pixels  →  array de formato (altura, largura, canais) = (3000, 4000, 3)
```

Em Python isso é um array **NumPy**. Em PyTorch vira um **tensor** (um array que pode morar na GPU), geralmente no formato `(canais, altura, largura)` e com valores normalizados (ex.: 0.0 a 1.0).

> Atenção: o OpenCV lê imagens em ordem **BGR**, não RGB. Isso é uma fonte clássica de bugs.

### As principais tarefas

| Tarefa | Pergunta que responde | Saída |
|---|---|---|
| **Classificação** | "O que tem na imagem?" | Um rótulo: `parede de escalada` |
| **Detecção de objetos** | "Onde estão os objetos?" | Caixas (bounding boxes) + classe + confiança |
| **Segmentação semântica** | "A que classe pertence cada pixel?" | Uma máscara por *classe*: todos os pixels "agarra" juntos |
| **Segmentação de instância** | "Quais pixels pertencem a *cada* objeto?" | Uma máscara por *objeto*: agarra 1, agarra 2, … |

```
Classificação      Detecção           Semântica          Instância
┌─────────┐       ┌─────────┐        ┌─────────┐        ┌─────────┐
│  ●   ▲  │       │ [●]  [▲]│        │  ■   ■  │        │  1   2  │
│    ■    │       │   [■]   │        │    ■    │        │    3    │
└─────────┘       └─────────┘        └─────────┘        └─────────┘
"parede"          3 caixas           "agarra" vs fundo  3 agarras separadas
```

## No nosso projeto

Queremos **clicar em uma agarra específica** e selecioná-la. Para isso precisamos:

- saber **quais pixels** são de cada agarra (o formato exato, não só uma caixa), e
- saber que a agarra A é **diferente** da agarra B, mesmo que tenham a mesma cor e estejam encostadas.

Isso é exatamente **segmentação de instância**. Uma caixa (detecção) seria imprecisa: agarras têm formatos irregulares e as caixas de agarras vizinhas se sobrepõem, então um clique poderia selecionar a errada.

Ambos os modelos que vamos usar produzem máscaras por instância:

- **SAM** → segmenta "tudo que parece um objeto", sem saber o que é agarra ([doc 3](03-segment-anything-sam.md))
- **YOLO-seg** → detecta e segmenta só a classe que treinarmos: agarra ([doc 4](04-yolo-segmentacao.md))

## Exemplo

```python
import cv2

img_bgr = cv2.imread("data/test/parede1.jpg")   # array (H, W, 3) em BGR
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(img_rgb.shape)     # (3000, 4000, 3)
print(img_rgb[100, 200]) # cor do pixel na linha 100, coluna 200 → ex. [212  45  38] (vermelho)
```

Repare: o índice é `[linha, coluna]` = `[y, x]`. Já as coordenadas de polígonos e caixas costumam vir como `(x, y)`. Outra fonte clássica de confusão.

## Para ir além

- [OpenCV — Getting Started with Images](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html)
- [NumPy — the absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Stanford CS231n — Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) (curso clássico, gratuito)
