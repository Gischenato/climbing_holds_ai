# 4. YOLO e segmentação

## O que é

**YOLO** ("You Only Look Once") é uma família de detectores de objetos famosa por ser **rápida**. O nome vem da ideia de olhar a imagem **uma única vez** (um único forward da rede) e já prever todos os objetos.

### One-stage vs. two-stage

- **Two-stage** (ex.: Mask R-CNN): primeiro propõe regiões candidatas, depois classifica e refina cada uma. Preciso, mas mais lento.
- **One-stage** (ex.: YOLO): prevê caixas e classes diretamente, em uma passada. Muito mais rápido, e hoje com precisão comparável.

### O que um detector prevê

Para cada objeto:

- **Bounding box**: retângulo `(x, y, largura, altura)` ou `(x1, y1, x2, y2)`
- **Classe**: ex. `hold`
- **Confiança** (score): de 0 a 1, o quão seguro o modelo está

A rede prevê *muitas* caixas candidatas, inclusive várias para o mesmo objeto. Uma etapa chamada **NMS** (Non-Maximum Suppression) remove as duplicadas — veja [Métricas](06-metricas.md#nms-non-maximum-suppression).

### YOLO-seg

A variante **-seg** adiciona uma **máscara** para cada objeto detectado. Ou seja: faz **segmentação de instância** ([doc 1](01-visao-computacional.md#as-principais-tarefas)) — exatamente o que precisamos para selecionar agarras com precisão.

Tamanhos de modelo (no Ultralytics YOLO11): `n` (nano) < `s` < `m` < `l` < `x`. Maior = mais preciso e mais lento. Começamos com `n` para iterar rápido.

### SAM vs. YOLO-seg

| | SAM (modo automático) | YOLO-seg treinado |
|---|---|---|
| Precisa treinar? | Não | Sim (fine-tuning) |
| Sabe o que é agarra? | Não (segmenta tudo) | Sim — só detecta o que aprendeu |
| Velocidade | Mais lento (encoder grande) | Muito rápido |
| Qualidade da borda da máscara | Excelente | Boa |
| Depende de dataset? | Não | Sim, e da qualidade das anotações |

## No nosso projeto

Na Fase 5 vamos:

1. Pegar um dataset público de agarras já anotado ([doc 5](05-datasets-e-anotacao.md)).
2. Fazer **fine-tuning** ([doc 2](02-redes-neurais-e-cnns.md#transfer-learning-e-fine-tuning)) de um `yolo11n-seg.pt` pré-treinado.
3. Comparar com o SAM usando as mesmas fotos de teste e as mesmas [métricas](06-metricas.md).
4. Plugar como `YoloDetector` no backend — o frontend nem precisa saber qual modelo foi usado.

## Exemplo

Com a biblioteca [Ultralytics](https://docs.ultralytics.com/):

```python
from ultralytics import YOLO

# Treino (fine-tuning)
model = YOLO("yolo11n-seg.pt")              # pesos pré-treinados no COCO
model.train(data="data/holds/data.yaml",    # aponta para o dataset
            epochs=100, imgsz=640, batch=16, device=0)

# Inferência
model = YOLO("runs/segment/train/weights/best.pt")
results = model("data/test/parede1.jpg", conf=0.25)

for r in results:
    polygons = r.masks.xy        # lista de polígonos em pixels da imagem original
    scores = r.boxes.conf        # confiança de cada agarra
```

O treino salva em `runs/segment/train/` gráficos de loss e mAP, exemplos de previsões e os pesos `best.pt` (melhor época na validação) e `last.pt`.

> ⚠️ **Licença**: o Ultralytics é **AGPL-3.0**. Para estudo e uso pessoal tudo bem; se o projeto virar um produto/serviço, é preciso abrir o código ou comprar licença comercial.

## Para ir além

- [Ultralytics — Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [Ultralytics — Train mode](https://docs.ultralytics.com/modes/train/)
- [Paper original YOLO (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — para entender a ideia que começou tudo
