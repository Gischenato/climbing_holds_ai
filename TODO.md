# TODO — Detector de Agarras de Escalada

**Objetivo do MVP:** subir a foto de uma parede de escalada → a IA encontra todas as agarras → eu clico nas agarras para selecionar/deselecionar.

**Estratégia:** começar com **SAM** (segmenta sem treino) e depois treinar um **YOLO-seg** específico para agarras.
**Stack:** Python + PyTorch (GPU) · FastAPI no backend · HTML/JS com `<canvas>` no frontend.

> Cada item tem um 📖 apontando para o documento que explica o conceito. Comece pela [trilha de estudo](docs/README.md).

---

## Fase 0 — Ambiente

- [x] Criar ambiente virtual Python (`python3 -m venv .venv`) e ativar
- [x] Instalar PyTorch **com CUDA** (pegar o comando certo em pytorch.org) — 📖 [GPU e CUDA](docs/02-redes-neurais-e-cnns.md#gpu-e-cuda)
- [x] Validar a GPU no WSL2: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [x] Criar a estrutura de pastas:
  ```
  backend/    # API FastAPI + detectores
  frontend/   # index.html, app.js, style.css
  models/     # checkpoints baixados/treinados (fora do git)
  data/       # imagens de teste e datasets (fora do git) — 📖 [organização](docs/05-datasets-e-anotacao.md#no-nosso-projeto)
  ```
- [x] `git init` + `.gitignore` (ignorar `.venv/`, `models/`, `data/`)
- [x] `requirements.txt` inicial

## Fase 1 — Detecção com SAM (sem treino)

- [ ] Estudar: 📖 [Visão computacional](docs/01-visao-computacional.md) e 📖 [Segment Anything](docs/03-segment-anything-sam.md)
- [x] Escolher a versão do SAM (**SAM 2.1 escolhido**: o SAM 3, que aceita prompt de texto como `"climbing hold"`, tem checkpoint restrito no HuggingFace — exige solicitar acesso; fica como opção futura)
- [x] Baixar o checkpoint para `models/`
- [x] Separar algumas fotos para testar (não precisa de anotação): imagens de `data/raw/roboflow-climbing-holds-v1/train/images/` servem para começar, mas fotos próprias em resolução cheia são melhores (as do dataset têm só 640 px)
- [x] Script `backend/try_sam.py`: carregar uma foto de parede e gerar todas as máscaras automaticamente
- [x] Salvar uma imagem com as máscaras sobrepostas para inspecionar o resultado
- [x] Ajustar parâmetros do gerador (`points_per_side`, `pred_iou_thresh`, `stability_score_thresh`) — 📖 [parâmetros](docs/03-segment-anything-sam.md#parâmetros-do-automatic-mask-generator)
  - Medido contra as anotações do dataset (24 fotos, 2318 agarras): `points_per_side=64`, `pred_iou_thresh=0.7`, `stability_score_thresh=0.85` → recall 58% → 80%, F1 0,65 → 0,73 (~7,7 s/foto na GPU). Ampliar a imagem 2x não ajudou.
- [x] Filtrar máscaras que não são agarras — 📖 [Pós-processamento](docs/07-mascaras-e-pos-processamento.md#filtragem)
  - [x] Área mínima / máxima (remove ruído e a parede inteira)
  - [x] Remover máscaras muito sobrepostas (ficar com a de melhor score)
  - [x] Remover máscaras que tocam a borda da imagem e são enormes (fundo)
- [x] Converter cada máscara em **polígono** (OpenCV `findContours` + `approxPolyDP`) — 📖 [máscara → polígono](docs/07-mascaras-e-pos-processamento.md#de-máscara-para-polígono)

## Fase 2 — Backend (FastAPI)

- [ ] Estudar: 📖 [Arquitetura da aplicação](docs/08-arquitetura-da-aplicacao.md)
- [x] Criar interface abstrata `Detector` com `detect(image) -> list[Hold]` (para trocar SAM ↔ YOLO depois sem mexer na API)
- [x] Implementar `SamDetector` com o código da Fase 1
- [x] Carregar o modelo **uma única vez** no startup (lifespan do FastAPI) — 📖 [por quê](docs/08-arquitetura-da-aplicacao.md#carregar-o-modelo-uma-vez)
- [x] Endpoint `POST /detect` recebendo a imagem (multipart/form-data)
- [x] Resposta JSON:
  ```json
  { "width": 4000, "height": 3000,
    "holds": [ { "id": 0, "polygon": [[x, y], ...], "bbox": [x, y, w, h], "score": 0.93 } ] }
  ```
- [x] Servir o `frontend/` como arquivos estáticos pelo próprio FastAPI
- [x] Testar com `curl` / Swagger em `http://localhost:8000/docs`

## Fase 3 — Frontend (selecionar/deselecionar)

- [x] Input de upload de imagem + botão "Detectar"
- [x] Exibir a imagem em um `<canvas>` ajustado à tela
- [x] Desenhar os polígonos das agarras por cima (contorno leve)
- [x] Converter coordenadas tela ↔ imagem — 📖 [coordenadas](docs/08-arquitetura-da-aplicacao.md#coordenadas-imagem--tela)
- [x] Clique → descobrir qual polígono contém o ponto (**hit-test**) — 📖 [ponto-em-polígono](docs/07-mascaras-e-pos-processamento.md#ponto-em-polígono-hit-testing)
- [x] Alternar seleção da agarra clicada (preenchimento destacado quando selecionada)
- [x] Se o clique cair em várias agarras sobrepostas, escolher a de **menor área**
- [x] Botões "Limpar seleção" e "Selecionar tudo" + contador de selecionadas
- [x] Indicador de "carregando" enquanto o backend processa

## Fase 4 — Avaliação do SAM

- [ ] Estudar: 📖 [Métricas](docs/06-metricas.md)
- [ ] Juntar ~20 fotos **próprias** de paredes variadas (ângulos, iluminação, cores) em `data/test/` — nenhuma pode estar nos datasets de treino
- [ ] Anotar manualmente as agarras das fotos (começar por ~5 para validar o critério e o script, depois o resto) — 📖 [Anotação](docs/05-datasets-e-anotacao.md)
  - Anotar também os **volumes (módulos)**, como classe separada: o SAM não distingue, mas o gabarito será usado para avaliar o YOLO na Fase 5
  - Classes do gabarito (v3): `hold` (0), `tape` (1, fitas que marcam as vias), `volume` (2). Exportado do Roboflow em YOLOv11 sem resize. Os índices mudam entre versões: o `evaluate.py` lê os nomes do `data.yaml`
  - Progresso: 21 fotos tiradas, 11 anotadas em `data/test/` (v3: 666 agarras, 60 volumes, 244 fitas)
- [x] Script de avaliação: precision, recall e IoU médio — `python -m backend.evaluate` (fitas e volumes do gabarito são ignorados, não contam como erro)
  - Resultado (6 fotos, 1024 px, IoU ≥ 0,5): precision 0,73 · recall 0,81 · F1 0,77 · IoU médio 0,90
  - Resultado (11 fotos, 1536 px): **precision 0,67 · recall 0,86 · F1 0,75 · IoU médio 0,90**
  - Em resolução cheia o SAM levava ~9 min por foto (devolve cada máscara no tamanho da entrada): o `SamDetector` agora reduz para 1024 px e escala os polígonos de volta
  - [x] Investigar a variação de tempo entre fotos (11 s a 278 s com o mesmo tamanho)
    - Causa: fragmentação do cache de memória do PyTorch (13 GB reservados para ~3 GB em uso, numa placa de 12 GB). No WSL2 a VRAM transborda para a RAM em vez de dar erro, e tudo fica lento a partir da 2ª foto
    - Correção: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` em `backend/__init__.py` → reserva estável em ~3,3 GB e **6–8 s por foto**
- [x] Anotar os principais erros (6 fotos):
  - **Falsos positivos**: quase todos **fora da parede** (teto, ventiladores, colchão, batente, a pessoa escalando). Ideia: detectar a região da parede e descartar o resto
  - **Agarras não achadas**: principalmente **agarras sobre volumes** (o SAM pega o volume e perde a agarra) e agarras **pequenas** em fotos panorâmicas (poucos pixels depois da redução para 1024 px)
  - Quando acha, o contorno é bom (IoU médio 0,90)
- [x] Testar resoluções maiores (`python -m backend.evaluate --max-side N`) — 📖 [resultados](docs/resultados-sam-resolucao.md)
  - 1536 px: recall 0,81 → 0,87 (agarras pequenas 57% → 69%), precision 0,73 → 0,63, ~10 s/foto. Acima de 1536 quase não muda. Confirmado nas 5 fotos novas da v2: recall 0,78 → 0,86, precision 0,79 → 0,71
- [x] Decidir o `max_side` padrão do `SamDetector` → **1536 px**
- [ ] Filtro para descartar máscaras fora da parede (maior fonte de falso positivo)
- [ ] Anotar as outras 10 fotos e rodar de novo

## Fase 4.1 — Preparar o dataset de treino (juntar datasets + caixas → polígonos)

> Os datasets públicos vêm em formatos diferentes: só **caixas** (linhas `classe x y w h`), só polígonos, ou misturados (o atual tem 8.756 caixas e 7.733 polígonos). Também vêm com classes diferentes (o atual tem 12: cores + `volume`). Treinar YOLO-seg com caixas ensina o modelo a prever retângulos.

- [ ] Juntar os datasets originais em `data/raw/<nome>/` (um por pasta, sem modificar) e conferir a licença de cada um
- [ ] Comparar o critério de anotação entre eles (volume conta? números de via? contorno justo?) — 📖 [regras de anotação](docs/05-datasets-e-anotacao.md#anotação)
- [ ] Classes finais: **`hold` (0) + `volume` (1)**; as cores ficam no original para o backlog
  - O dataset atual tem só 89 volumes contra ~16.400 agarras. Manter a classe mesmo assim: é fácil filtrar depois (gerar um dataset só de agarras ou só de volumes), e o contrário exige reanotar
- [ ] Script `backend/prepare_dataset.py` (lê um ou mais datasets → gera `data/holds/`):
  - [ ] Remapear as classes de cada dataset para as classes finais (descartar o que não é agarra)
  - [ ] Linhas com **caixa**: converter para pixels e passar como **prompt de caixa** para o SAM — 📖 [tipos de prompt](docs/03-segment-anything-sam.md#tipos-de-prompt)
  - [ ] Converter a máscara em polígono (reusar `mask_to_polygon` da Fase 1) — 📖 [máscara → polígono](docs/07-mascaras-e-pos-processamento.md#de-máscara-para-polígono)
  - [ ] Linhas que já são **polígono**: manter
  - [ ] Normalizar para 0–1 e gravar no formato **YOLO segmentation** — 📖 [formato](docs/05-datasets-e-anotacao.md#formatos-comuns)
  - [ ] Remover imagens duplicadas entre datasets (forks no Roboflow são comuns) — por hash
  - [ ] Criar o split train / val (~85/15); o test são as fotos próprias da Fase 4
  - [ ] Gerar um `data.yaml` novo com caminhos e classes corretos
  - [ ] Salvar imagens de debug com os polígonos desenhados (gerados pelo SAM em outra cor)
- [ ] Revisar uma amostra, com atenção a agarras sobre volumes e agarras vizinhas grudadas
- [ ] Lembrar: agarras sem anotação no dataset original continuam sem anotação

## Fase 5 — YOLO-seg treinado para agarras

- [ ] Estudar: 📖 [Redes neurais e transfer learning](docs/02-redes-neurais-e-cnns.md), 📖 [YOLO](docs/04-yolo-segmentacao.md), 📖 [Datasets](docs/05-datasets-e-anotacao.md)
- [ ] Buscar mais datasets públicos de agarras (Roboflow Universe, busca "climbing holds"), exportar em **YOLOv11** e passar pela [Fase 4.1](#fase-41--preparar-o-dataset-de-treino-juntar-datasets--caixas--polígonos)
- [ ] Conferir o `data/holds/data.yaml` gerado e o split train / val
- [ ] Fine-tuning do `yolo11n-seg.pt` (começar pequeno, depois testar `s`/`m`)
- [ ] Analisar as curvas de treino (loss, mAP) e checar overfitting
- [ ] Avaliar no **mesmo** conjunto de teste da Fase 4 e comparar com o SAM
  - Olhar o mAP **por classe** (`hold` e `volume` separados)
- [ ] Repetir o teste de resolução com o YOLO: inferência com `imgsz` 640 / 960 / 1280 (e treino no mesmo tamanho), olhando o recall das agarras **pequenas** — comparar com a tabela do SAM em 📖 [resultados](docs/resultados-sam-resolucao.md)
  - O `evaluate.py` já mede tudo isso; só falta o `--max-side` virar um parâmetro genérico do detector
- [ ] Se volumes ficarem ruins: primeiro buscar **mais exemplos** de volume (outros datasets, fotos próprias de paredes com muitos módulos)
- [ ] Implementar `YoloDetector` e permitir escolher o detector (config ou parâmetro da API)
  - Adicionar `kind: "hold" | "volume"` ao `Hold` e à resposta do `/detect`; o frontend desenha volumes com outra cor (atualizar `docs/api.md`)
- [ ] Opcional: dois modelos separados (um de agarras, outro de volumes) combinados por um `CombinedDetector` **no backend**, devolvendo uma lista só. Vale a pena se quiser iterar no de volumes sem retreinar o de agarras, ou rodar volumes numa resolução menor
- [ ] ⚠️ Ultralytics é licença **AGPL-3.0** — ok para estudo; ver implicações se o projeto virar produto

## Backlog (futuro)

- [ ] Classificar o tipo de agarra (jug, crimp, sloper, pinch, pocket, volume…)
- [ ] Agrupar agarras por cor (para identificar vias/boulders)
- [ ] Usar as **fitas** (`tape`, já anotadas no gabarito da Fase 4) para achar as agarras de cada via
- [ ] Salvar boulders (conjunto de agarras selecionadas + nome + grau)
- [ ] Marcar agarras de início / fim / só pé
- [ ] Exportar imagem com a seleção destacada
- [ ] Ajuste manual: adicionar agarra que a IA não achou / remover falso positivo
- [ ] Usar os ajustes manuais como novos dados de treino (ciclo de melhoria)
- [ ] Layout para celular (tirar foto direto na academia)
