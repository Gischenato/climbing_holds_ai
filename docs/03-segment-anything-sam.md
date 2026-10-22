# 3. Segment Anything (SAM)

## O que é

O **SAM** (Segment Anything Model), da Meta, é um **foundation model** de segmentação: foi treinado em um volume enorme de dados (o dataset SA-1B tem ~11 milhões de imagens e mais de 1 bilhão de máscaras) para segmentar **qualquer objeto**, mesmo de tipos que nunca viu.

Isso se chama **zero-shot**: funciona em uma tarefa nova sem nenhum treino adicional. É por isso que começamos por ele — dá para ver resultado no primeiro dia.

### Versões

| Versão | Destaque |
|---|---|
| **SAM** (2023) | Imagens; prompts de ponto, caixa e máscara |
| **SAM 2** (2024) | Mais rápido e preciso; também segmenta vídeo |
| **SAM 3** (2025) | Aceita **prompt de conceito em texto** (ex.: `"climbing hold"`) e devolve todas as instâncias daquele conceito |

> Confira a versão mais recente e o modo de instalação no repositório oficial antes de implementar — essas bibliotecas evoluem rápido.

O SAM 3 é especialmente interessante para nós: em vez de "segmente tudo e depois filtre", podemos pedir "segmente todas as agarras".

### Arquitetura (simplificada)

```
                    ┌──────────────────┐
 imagem ──────────▶ │  Image encoder   │ ──▶ embedding da imagem  (pesado, roda 1x por imagem)
                    └──────────────────┘            │
                    ┌──────────────────┐            ▼
 prompt ──────────▶ │  Prompt encoder  │ ──▶ ┌──────────────┐
 (ponto/caixa/texto)└──────────────────┘     │ Mask decoder │ ──▶ máscaras + score
                                             └──────────────┘  (leve, roda por prompt)
```

- **Image encoder**: um Transformer de visão grande que transforma a imagem em um "resumo" numérico (embedding). É a parte lenta.
- **Prompt encoder**: transforma o que você indica (um clique, uma caixa, um texto) em números.
- **Mask decoder**: combina os dois e gera a máscara. É rápido, então dá para testar muitos prompts sobre a mesma imagem.

### Tipos de prompt

- **Ponto**: "segmente o objeto que está aqui" (clique). Pode ter pontos positivos e negativos.
- **Caixa**: "segmente o objeto dentro desta caixa".
- **Texto / conceito** (SAM 3): "segmente todas as coisas que são X".
- **Nenhum** → modo **automático**.

### Automatic Mask Generator

O modo automático coloca uma **grade de pontos** sobre a imagem (ex.: 32×32 = 1024 pontos), usa cada ponto como prompt, e depois limpa o resultado: remove máscaras de baixa qualidade e duplicadas.

Cada máscara gerada vem com campos como:

| Campo | Significado |
|---|---|
| `segmentation` | máscara binária (H×W de True/False) |
| `area` | número de pixels da máscara |
| `bbox` | caixa `[x, y, largura, altura]` |
| `predicted_iou` | quão boa o modelo *acha* que a máscara é |
| `stability_score` | quão estável a máscara é se mudarmos levemente o limiar |

### Parâmetros do Automatic Mask Generator

| Parâmetro | O que faz | Efeito para agarras |
|---|---|---|
| `points_per_side` | Densidade da grade de pontos | Mais pontos → acha agarras pequenas, mas fica mais lento |
| `pred_iou_thresh` | Descarta máscaras com `predicted_iou` abaixo disso | Subir → menos lixo, mas pode perder agarras |
| `stability_score_thresh` | Descarta máscaras instáveis | Subir → máscaras mais "limpas" |
| `min_mask_region_area` | Remove pedacinhos e buracos pequenos | Útil para limpar ruído |
| `crop_n_layers` | Roda também em recortes ampliados da imagem | Ajuda com agarras pequenas em fotos grandes |

## No nosso projeto

O SAM (modo automático) não sabe o que é uma agarra — ele segmenta **tudo**: agarras, mas também parafusos, painéis da parede, sombras, o chão, pessoas. Por isso a Fase 1 inclui **filtragem** ([doc 7](07-mascaras-e-pos-processamento.md#filtragem)).

Limitações esperadas:

- **Volumes** (grandes formas na parede) podem ser confundidos com a parede, ou agarras sobre volumes podem ser engolidas pela máscara do volume.
- **Agarras pequenas** (pés, crimps) podem não ser detectadas com grade pouco densa.
- **Agarras da mesma cor da parede** (comum em paredes muito usadas, cheias de magnésio) são difíceis.
- Fotos muito grandes são redimensionadas internamente; detalhes podem se perder.

Essas limitações são a motivação para a Fase 5 (YOLO treinado especificamente em agarras).

## Exemplo

Pseudocódigo do fluxo (a API exata depende da versão escolhida):

```python
# 1. Carregar o modelo uma vez (na GPU)
model = carregar_sam("models/sam_checkpoint.pt", device="cuda")
generator = AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.8)

# 2. Gerar as máscaras
masks = generator.generate(img_rgb)          # lista de dicts
print(len(masks), "máscaras encontradas")

# 3. Filtrar o que não parece agarra
area_total = img_rgb.shape[0] * img_rgb.shape[1]
holds = [m for m in masks
         if 0.0002 * area_total < m["area"] < 0.05 * area_total]
```

## Para ir além

- [Paper original — "Segment Anything" (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- [Repositório SAM 2](https://github.com/facebookresearch/sam2)
- [Página do projeto Segment Anything (Meta AI)](https://ai.meta.com/research/sam/)
- Notebook `automatic_mask_generator_example.ipynb` no repositório do SAM 2 (pasta `notebooks/`)
