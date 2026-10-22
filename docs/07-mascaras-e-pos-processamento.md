# 7. Máscaras e pós-processamento

## O que é

O modelo devolve máscaras "cruas". Antes de mandar para o frontend, precisamos **limpar** (filtrar o que não é agarra) e **converter** para um formato leve e fácil de desenhar e clicar.

### Máscara binária

Uma **máscara** é um array do mesmo tamanho da imagem, com `True` onde está o objeto e `False` no resto.

```
. . . . . . . .
. . # # # . . .      # = True (pixel da agarra)
. # # # # # . .      . = False
. . # # # # . .
. . . # # . . .
. . . . . . . .
```

Problema: uma máscara de uma foto 4000×3000 tem 12 milhões de valores. Com 200 agarras, isso é enorme para mandar ao navegador.

### Representações compactas

| Formato | Ideia | Uso |
|---|---|---|
| **Máscara binária** | Um valor por pixel | Processamento no backend |
| **RLE** (Run-Length Encoding) | Guarda "sequências": 50 False, 12 True, 30 False… | Armazenar/transferir máscaras exatas (COCO usa) |
| **Polígono** | Lista de pontos `(x, y)` do contorno | Desenhar e clicar no frontend ✅ |

Vamos usar **polígonos**: algumas dezenas de pontos por agarra, fáceis de desenhar em um `<canvas>` e de testar cliques.

## De máscara para polígono

Com OpenCV:

1. **`cv2.findContours`** encontra o contorno (a borda) das regiões `True`.
2. **`cv2.approxPolyDP`** simplifica o contorno, removendo pontos desnecessários (algoritmo de Douglas–Peucker). O parâmetro `epsilon` controla quanto simplificar: maior = menos pontos, menos precisão.

```python
import cv2
import numpy as np

def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.002):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)          # fica com a maior parte
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2).tolist()                 # [[x, y], [x, y], ...]
```

`RETR_EXTERNAL` pega só o contorno de fora (ignora buracos — ótimo para agarras com furo de parafuso no meio).

## Filtragem

O SAM segmenta tudo. Regras simples para ficar só com o que parece agarra:

| Filtro | Remove | Como |
|---|---|---|
| **Área mínima** | Ruído, parafusos, pontinhos | `area < X% da imagem` → descarta |
| **Área máxima** | A parede inteira, painéis, chão | `area > Y% da imagem` → descarta |
| **Score** | Máscaras de baixa qualidade | `predicted_iou` ou `stability_score` abaixo do limiar |
| **Toca a borda + grande** | Fundo, teto, chão | bbox encostada na borda da imagem e área grande |
| **Sobreposição** | Duplicatas (mesma agarra 2x) | Se IoU entre duas máscaras > 0.7, fica a de maior score (tipo NMS) |
| **Contém outras** | Volume que "engole" as agarras em cima dele | Se uma máscara contém várias outras menores, avaliar descartar |
| **Formato** (opcional) | Faixas finas, linhas | Razão de aspecto da bbox muito extrema |

Os valores de X e Y dependem da distância da foto. Comece com algo como **0.02%** e **5%** e ajuste olhando os resultados.

> Dica: salve uma imagem de debug com as máscaras **descartadas** em vermelho e as **mantidas** em verde. Isso torna o ajuste dos filtros muito mais fácil.

## Ponto-em-polígono (hit-testing)

Quando o usuário clica, precisamos descobrir **qual agarra** contém o ponto do clique.

Algoritmo clássico **ray casting**: lance um raio horizontal do ponto para a direita e conte quantas vezes ele cruza as arestas do polígono. **Ímpar = dentro, par = fora.**

```
        ┌───────┐
   ●────┼───────┼──▶   cruza 2 vezes → fora
        │   ●───┼──▶   cruza 1 vez   → dentro
        └───────┘
```

No navegador nem precisamos implementar: o canvas já faz isso.

```js
// path = new Path2D() montado com os pontos do polígono
ctx.isPointInPath(path, x, y)  // true se (x, y) está dentro
```

Se o clique cair dentro de **mais de uma** agarra (máscaras sobrepostas), escolhemos a de **menor área** — normalmente é a agarra pequena em cima de um volume.

## No nosso projeto

```
máscaras do SAM ──▶ filtros ──▶ mask_to_polygon ──▶ JSON { polygon, bbox, score } ──▶ frontend
```

- Fase 1: filtragem + conversão para polígono
- Fase 3: hit-testing do clique no frontend

## Para ir além

- [OpenCV — Contours: Getting Started](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [OpenCV — Contour Features (approxPolyDP, área, bbox)](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
- [MDN — CanvasRenderingContext2D.isPointInPath](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/isPointInPath)
- [Wikipedia — Point in polygon](https://en.wikipedia.org/wiki/Point_in_polygon)
