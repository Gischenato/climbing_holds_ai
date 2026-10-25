# Resultados — SAM em resoluções maiores (Fase 4)

> **Decisão:** o padrão do `SamDetector` passou a ser **1536 px** depois deste experimento. Repetir a comparação de resolução com o YOLO na Fase 5 (ver o [TODO](../TODO.md#fase-5--yolo-seg-treinado-para-agarras)).

**Pergunta:** reduzir as fotos para 1024 px antes do SAM está fazendo a gente perder agarras? Vale a pena usar uma resolução maior?

**Resposta curta:** sim. Com 1536 px, o SAM acha **5,5 pontos a mais de agarras** (recall de 81% → 87%), principalmente as pequenas (57% → 69%), e cada foto leva 10 s em vez de 7 s. O custo é a precision, que cai de 73% para 63%, porque ele também passa a marcar furos, parafusos e marcas na parede. Subir além de 1536 px quase não ajuda.

---

## Como foi medido

- **Gabarito:** as 6 fotos anotadas em `data/test/` (326 agarras; fitas e volumes são ignorados, como combinado).
- **Acerto:** uma previsão casa com uma agarra quando o IoU das máscaras é ≥ 0,5 (ver [docs/06-metricas.md](06-metricas.md)).
- **Mesmos parâmetros do SAM e dos filtros em todos os testes**, mudando só:
  - `max_side`: o lado maior da imagem entregue ao SAM (a foto original tem 4000 px);
  - `crops`: com `crop_n_layers=1` o SAM roda na foto inteira **e** em 4 pedaços dela, cada um ampliado. É a forma "oficial" do SAM de ver objetos pequenos com mais detalhe.
- **Tamanho das agarras:** separei as 326 agarras em três faixas do mesmo tamanho pelo lado equivalente (raiz da área) na foto original: **pequena** (< 60 px), **média** (60–117 px) e **grande** (≥ 117 px).

Para repetir uma linha: `python -m backend.evaluate --max-side 1024` (sem `--max-side` usa o padrão, 1536).

## Resultados

| Configuração | Precision | Recall | F1 | IoU médio | Tempo/foto (média · máx) | VRAM |
|---|---|---|---|---|---|---|
| **1024** (padrão antigo) | **0,73** | 0,81 | **0,77** | 0,90 | **6,7 s** · 8,6 s | 3,4 GB |
| **1536** (padrão novo) | 0,63 | 0,87 | 0,73 | 0,91 | 10,1 s · 19 s | 3,4 GB |
| 2048 | 0,62 | 0,87 | 0,73 | 0,91 | 14,9 s · 33 s | 4,5 GB |
| 3000 | 0,59 | 0,88 | 0,71 | 0,91 | 26,4 s · 69 s | 7,4 GB |
| 1024 + crops | 0,65 | 0,88 | 0,75 | 0,89 | 24,2 s · 30 s | 3,4 GB |
| 2048 + crops | 0,56 | **0,90** | 0,69 | 0,91 | 35,6 s · 59 s | 4,5 GB |

Em números absolutos (de 326 agarras):

| Configuração | Acertos (TP) | Falsos positivos (FP) | Não achadas (FN) |
|---|---|---|---|
| 1024 | 265 | 99 | 61 |
| 1536 | 283 | 164 | 43 |
| 2048 | 285 | 174 | 41 |
| 3000 | 286 | 198 | 40 |
| 1024 + crops | 288 | 154 | 38 |
| 2048 + crops | 292 | 225 | 34 |

### Recall por tamanho de agarra

| Configuração | Pequena (108) | Média (107) | Grande (111) |
|---|---|---|---|
| 1024 | **57%** | 91% | 95% |
| 1536 | 69% | 95% | 95% |
| 2048 | 71% | 94% | 96% |
| 3000 | 72% | 95% | 95% |
| 1024 + crops | 74% | 95% | 95% |
| 2048 + crops | 75% | 98% | 95% |

## O que os números dizem

### 1. O problema do recall está nas agarras pequenas

Agarras grandes já são achadas 95% das vezes em qualquer configuração. Quase todo o ganho vem das **pequenas** (pés, crimps pequenos), que vão de 57% para ~70%.

### 2. O SAM não acha essas agarras, e o problema não é o contorno

Seria possível que, em 1024 px, o SAM achasse a agarra pequena mas com o contorno tão "pixelado" que o IoU ficasse abaixo de 0,5. Medi isso: para cada agarra não achada, procurei a previsão mais parecida.

| | Agarras pequenas não achadas | Sem nenhuma máscara por perto (IoU < 0,1) | Achou, mas contorno ruim (IoU 0,1–0,5) |
|---|---|---|---|
| 1024 | 46 | 42 | 4 |
| 2048 | 31 | 28 | 3 |

Ou seja, ele simplesmente não as encontra.

Isso é curioso, porque o "cérebro" do SAM (o *image encoder*) **sempre** trabalha com a imagem em 1024 px, não importa o tamanho que a gente entrega. O que muda com a resolução de entrada é o que acontece **depois**: o SAM devolve cada máscara no tamanho da imagem recebida e só então aplica os filtros de qualidade (`stability_score`, `min_mask_region_area=50` px…). Em 1024 px, uma agarra pequena tem só ~15×15 pixels, e é provável que um desses filtros a descarte por ter poucos pixels. Isso é uma **hipótese**: não verifiquei qual filtro é o responsável.

### 3. Mais resolução = mais falsos positivos

Os FPs vão de 99 para 164 com 1536 px. Comparando as imagens de 1024 e 2048 (`data/out/eval/` e `data/out/eval-2048/`), os novos falsos positivos são **coisas pequenas na própria parede**: furos de parafuso, parafusos, marcas de sola. Também aparecem mais pedaços de coisas fora da parede (teto, colchão). É o mesmo efeito do item 2 ao contrário: o SAM passa a enxergar tudo o que é pequeno, não só agarras.

### 4. Depois de 1536 px, quase nada muda

De 1536 para 3000 px: só +3 acertos, +34 falsos positivos e 2,6× mais tempo. Com 326 agarras, cada agarra vale ~0,3 ponto de recall, então as diferenças entre 1536, 2048 e 3000 estão dentro do ruído.

### 5. Os crops dão o maior recall, mas custam caro

`1024 + crops` tem recall parecido com 3000 px (88%), com menos falsos positivos e menos VRAM, mas leva 24 s por foto (o SAM roda 5 vezes). `2048 + crops` é o recall mais alto (90%), mas com 35 s por foto e a pior precision.

### 6. O contorno não é o problema

O IoU médio dos acertos fica em ~0,90 em todas as configurações: quando o SAM acha uma agarra, o contorno é bom.

## Recomendação

**Mudar o padrão para 1536 px.** Para o app, o **recall importa mais** que a precision: uma agarra que a IA não acha não pode ser selecionada, enquanto um falso positivo o usuário simplesmente não clica (ver [docs/06-metricas.md](06-metricas.md#precision-recall-e-f1)). 1536 px dá +18 agarras achadas por +3,4 s por foto, sem VRAM extra.

A queda da precision provavelmente dá para recuperar com filtros:

1. **Descartar o que está fora da parede**: era a maior fonte de FP já em 1024 px (teto, ventiladores, colchão, a pessoa escalando).
2. **Descartar furos e parafusos**: são pequenos, quase circulares e escuros. Um filtro de área mínima maior junto com cor/formato pode resolver, mas cuidado para não cortar as agarras pequenas que acabamos de ganhar.

Os crops ficam como opção se o tempo deixar de importar (por exemplo, processar a foto uma vez e guardar o resultado).

## Validação com fotos novas

Depois da decisão, a v2 do gabarito trouxe **5 fotos novas** (340 agarras) que não foram usadas para escolher a resolução. Elas servem para checar se a conclusão não foi "decorada" nas 6 primeiras:

| Só as 5 fotos novas | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| 1024 | 0,79 | 0,78 | 266 | 69 | 74 |
| 1536 | 0,71 | 0,86 | 291 | 120 | 49 |

O padrão se repete: com 1536 px o recall sobe (+7 pontos) e a precision cai (−9 pontos). A decisão se mantém. Nas 11 fotos juntas, 1536 px dá **precision 0,67 · recall 0,86 · F1 0,75**.

## Limitações

- **Poucas fotos** (6 na tabela principal, 11 na validação), todas da mesma academia e do mesmo dia.
- Os parâmetros do SAM e dos filtros foram ajustados na Fase 1 com o dataset do Roboflow em **640 px**. Com 1536 px, talvez valha reajustá-los ([`backend/tune_filters.py`](../backend/tune_filters.py)).
- O tempo é numa RTX 4070 SUPER no WSL2. O máximo varia mais que a média porque a foto panorâmica gera mais máscaras.

## Próximos passos sugeridos

1. ~~Decidir o `max_side` padrão do app~~ → 1536 px.
2. Anotar as outras 15 fotos e repetir esta tabela.
3. Testar um filtro "fora da parede" para recuperar precision.
4. (Opcional) Descobrir qual filtro do SAM descarta as agarras pequenas em 1024 px, testando `min_mask_region_area` e `stability_score_thresh` mais baixos.
