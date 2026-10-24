# Docs — material de estudo

Estes documentos explicam os conceitos usados no projeto do detector de agarras. A ideia não é ser um curso completo, mas dar a **intuição** de cada coisa, mostrar **onde ela aparece no nosso projeto** e apontar para onde aprofundar.

Cada documento segue a mesma estrutura:

1. **O que é** — explicação intuitiva
2. **No nosso projeto** — como o conceito se aplica às agarras
3. **Exemplo** — código curto ou pseudocódigo
4. **Para ir além** — links oficiais

## Trilha de estudo

Ordem sugerida, ligada às fases do [TODO](../TODO.md):

| # | Documento | Ler antes da… |
|---|---|---|
| 1 | [Visão computacional: imagens e tarefas](01-visao-computacional.md) | Fase 1 |
| 2 | [Redes neurais, CNNs e transfer learning](02-redes-neurais-e-cnns.md) | Fase 0 (parte de GPU) e Fase 5 |
| 3 | [Segment Anything (SAM)](03-segment-anything-sam.md) | Fase 1 |
| 4 | [Máscaras e pós-processamento](07-mascaras-e-pos-processamento.md) | Fases 1 e 3 |
| 5 | [Arquitetura da aplicação](08-arquitetura-da-aplicacao.md) | Fases 2 e 3 |
| 6 | [Métricas de avaliação](06-metricas.md) | Fase 4 |
| 7 | [Datasets e anotação](05-datasets-e-anotacao.md) | Fases 4 e 5 |
| 8 | [YOLO e segmentação](04-yolo-segmentacao.md) | Fase 5 |

Resultados de experimentos: [SAM em resoluções maiores](resultados-sam-resolucao.md) (Fase 4).

Referência do que o backend expõe (endpoints, campos, erros): [Referência da API](api.md).

Dúvida rápida sobre um termo? Veja o [glossário](glossario.md).

## Mapa mental do projeto

```
 Foto da parede
       │
       ▼
 ┌─────────────┐   upload (HTTP)   ┌──────────────────────────────┐
 │  Frontend   │ ────────────────▶ │  Backend FastAPI             │
 │  (canvas)   │                   │   Detector (SAM ou YOLO)     │
 │             │ ◀──────────────── │   → máscaras                 │
 │ clique =    │   JSON polígonos  │   → filtragem                │
 │ seleciona   │                   │   → polígonos                │
 └─────────────┘                   └──────────────────────────────┘
```
