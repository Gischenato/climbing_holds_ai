# 6. Métricas de avaliação

## O que é

Para saber se o detector é bom (e se o YOLO é melhor que o SAM), precisamos de **números**, não só de "parece bom". As métricas comparam as previsões com o gabarito anotado.

### IoU (Intersection over Union)

Mede o quanto duas regiões se sobrepõem:

```
          área da interseção
IoU = ─────────────────────────
           área da união

 ┌──────┐
 │  A ┌─┼────┐        IoU = 0   → nada em comum
 │    │█│    │        IoU = 0.5 → sobreposição razoável
 └────┼─┘  B │        IoU = 1   → idênticas
      └──────┘
```

Vale para caixas e para máscaras (contando pixels). Usamos o IoU para decidir se uma previsão "acertou" uma agarra anotada: normalmente **IoU ≥ 0.5** conta como acerto.

### Verdadeiro/falso positivo e falso negativo

| | Significado | No nosso caso |
|---|---|---|
| **TP** (verdadeiro positivo) | Previu e existe | Achou uma agarra real |
| **FP** (falso positivo) | Previu mas não existe | Marcou um parafuso ou sombra como agarra |
| **FN** (falso negativo) | Existe mas não previu | Deixou de achar uma agarra |

### Precision, recall e F1

```
precision = TP / (TP + FP)    → "das agarras que o modelo marcou, quantas eram agarras?"
recall    = TP / (TP + FN)    → "das agarras reais, quantas o modelo achou?"
F1        = 2 · P · R / (P + R)  → média harmônica, um número só
```

Existe um **trade-off** controlado pelo **limiar de confiança** (`conf`):

- `conf` alto → só previsões muito seguras → **precision alta, recall baixo**
- `conf` baixo → aceita quase tudo → **recall alto, precision baixa**

Para o nosso app, **recall importa muito**: se a IA não acha uma agarra, o usuário não consegue selecioná-la. Um falso positivo é menos grave (o usuário simplesmente não clica nele).

### mAP (mean Average Precision)

A métrica padrão de detectores, que o Ultralytics mostra no treino:

1. Varia o limiar de confiança e traça a **curva precision × recall**.
2. **AP** = área sob essa curva (um número de 0 a 1) para uma classe.
3. **mAP** = média dos APs de todas as classes (nós temos só uma: `hold`).

Variações:

- **mAP@50**: considera acerto com IoU ≥ 0.5 — "achou a agarra mais ou menos no lugar certo?"
- **mAP@50-95**: média de mAP com IoU de 0.5 a 0.95 — exige **bordas precisas**. Mais rigoroso.

No Ultralytics, para segmentação aparecem métricas de `Box` e de `Mask` separadas. A de **Mask** é a que mais importa para nós.

### NMS (Non-Maximum Suppression)

Detectores geram várias previsões para o mesmo objeto. O NMS limpa isso:

1. Ordena as previsões por confiança.
2. Pega a melhor e remove todas as outras com IoU acima de um limiar (ex.: 0.7) em relação a ela.
3. Repete com as que sobraram.

Faremos algo parecido manualmente com as máscaras do SAM, para remover duplicatas ([doc 7](07-mascaras-e-pos-processamento.md#filtragem)).

## No nosso projeto

**Fase 4** — script de avaliação simples:

```python
def avaliar(previstas, anotadas, iou_min=0.5):
    usadas = set()
    tp = 0
    for p in sorted(previstas, key=lambda m: -m.score):
        melhor = max(
            (a for a in range(len(anotadas)) if a not in usadas),
            key=lambda a: iou(p.mask, anotadas[a].mask),
            default=None,
        )
        if melhor is not None and iou(p.mask, anotadas[melhor].mask) >= iou_min:
            usadas.add(melhor)
            tp += 1
    fp = len(previstas) - tp
    fn = len(anotadas) - tp
    return tp / (tp + fp), tp / (tp + fn)   # precision, recall

def iou(a, b):  # a, b: máscaras booleanas numpy (H, W)
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union else 0.0
```

O script real é o [backend/evaluate.py](../backend/evaluate.py) (`python -m backend.evaluate`). Diferenças em relação ao esboço acima:

- Avalia qualquer `Detector`, então serve igual para o SAM e o YOLO.
- O gabarito tem as classes `hold`, `tape` (fitas das vias) e `volume`. Uma previsão que casa com fita ou volume é **ignorada**: não conta como acerto nem como erro, só é contada à parte. O recall considera só as `hold`.
- As máscaras são recortadas no bbox de cada polígono; máscaras inteiras em 4000×1848 não caberiam na memória.
- Salva em `data/out/eval/` cada foto com verde = acerto, vermelho = agarra não achada, amarelo = falso positivo, azul = ignorada.

**Fase 5** — comparar SAM vs. YOLO no **mesmo** conjunto de teste, com o mesmo script.

## Para ir além

- [Ultralytics — guia de métricas de performance](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [COCO — detection evaluation](https://cocodataset.org/#detection-eval)
