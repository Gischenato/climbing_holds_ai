# 5. Datasets e anotação

## O que é

Um **dataset** é um conjunto de imagens + **anotações** (o gabarito feito por humanos). Para segmentação de instância, cada anotação diz: "nesta imagem existe um objeto da classe X com este formato (polígono)".

O modelo só fica tão bom quanto os dados: anotações erradas ou inconsistentes (agarras esquecidas, polígonos tortos) ensinam erros ao modelo.

### Formatos comuns

**YOLO segmentation** — um `.txt` por imagem, uma linha por objeto. Coordenadas **normalizadas** (0 a 1, divididas pela largura/altura da imagem):

```
# classe x1 y1 x2 y2 x3 y3 ...
0 0.412 0.301 0.425 0.298 0.431 0.315 0.418 0.322
0 0.705 0.640 0.722 0.633 0.730 0.661 0.709 0.668
```

Mais um `data.yaml` descrevendo o dataset:

```yaml
path: data/holds
train: images/train
val: images/val
test: images/test
names:
  0: hold
```

**COCO JSON** — um único JSON com todas as imagens e anotações; polígonos em pixels absolutos (ou máscaras em RLE). É o formato do dataset COCO e muito usado em pesquisa.

Ferramentas como o Roboflow convertem entre formatos.

### Anotação

Ferramentas para desenhar polígonos: **Roboflow Annotate**, **CVAT**, **Label Studio**. Muitas têm "smart polygon" usando o próprio SAM: você clica na agarra e ela gera o polígono — muito mais rápido que desenhar à mão.

Regras para anotar agarras de forma consistente:

- Anotar **todas** as agarras visíveis da imagem (uma agarra esquecida vira "fundo" no treino e ensina o modelo a ignorá-la).
- Decidir antes: volume é agarra? Agarra parcialmente coberta conta? Parafuso não conta.
- Contornar a borda real da agarra, não a sombra.

### Train / validation / test

O dataset é dividido em 3 partes:

| Split | Uso | Proporção típica |
|---|---|---|
| **Train** | O modelo aprende com essas imagens | ~70–80% |
| **Validation** | Avalia durante o treino; escolhe a melhor época | ~10–20% |
| **Test** | Avaliação final, **só no fim** | ~10% |

Regra de ouro: o modelo **nunca** pode ver as imagens de teste no treino. Cuidado com fotos quase iguais (mesma parede, ângulo ligeiramente diferente) em splits diferentes — isso infla as métricas.

### Overfitting

**Overfitting** é quando o modelo "decora" o treino em vez de aprender o padrão geral. Sintoma: a loss de **treino** continua caindo, mas a de **validação** para de cair ou sobe.

```
loss
 │\
 │ \_____________ validação começa a subir → overfitting
 │  \      ___/
 │   \____/
 │        \______ treino continua caindo
 └──────────────────── épocas
```

Remédios: mais dados, **data augmentation**, modelo menor, menos épocas (*early stopping*).

### Data augmentation

Gerar variações das imagens de treino para o modelo ver mais diversidade: espelhar, rotacionar levemente, mudar brilho/contraste/saturação, recortar, juntar 4 imagens em uma (*mosaic*). O Ultralytics já aplica várias automaticamente.

Para agarras, cuidado com **mudanças de cor (hue)** muito fortes se no futuro quisermos agrupar agarras por cor.

## No nosso projeto

- **Fase 4**: anotamos à mão um pequeno conjunto de teste (nossas próprias fotos) para medir o SAM. Ele também servirá para comparar com o YOLO.
- **Fase 5**: usamos um dataset público (busque "climbing holds" no [Roboflow Universe](https://universe.roboflow.com/)), conferindo licença, quantidade de imagens e se as anotações são **polígonos** (não só caixas).
- **Futuro**: as correções manuais feitas no app (adicionar/remover agarra) podem virar novas anotações — um ciclo de melhoria contínua.

Organização da pasta `data/` (fora do git):

```
data/
  raw/                           # datasets públicos originais, sem modificar
    roboflow-climbing-holds-v1/  # um por pasta: <origem>-<nome>-v<versão>
  test/                          # fotos próprias anotadas (Fase 4), nunca usadas no treino
    images/  labels/  data.yaml  # export YOLOv11 do Roboflow, sem resize
  holds/                         # dataset final gerado pelo prepare_dataset.py (Fase 4.1)
  out/                           # imagens de debug dos scripts
```

## Para ir além

- [Ultralytics — formato de dataset de segmentação](https://docs.ultralytics.com/datasets/segment/)
- [COCO — formato de dados](https://cocodataset.org/#format-data)
- [CVAT](https://www.cvat.ai/) · [Label Studio](https://labelstud.io/) · [Roboflow](https://roboflow.com/)
