# Glossário

| Termo | Significado |
|---|---|
| **Anotação** | Gabarito feito por humanos: onde estão os objetos em uma imagem. [doc 5](05-datasets-e-anotacao.md) |
| **Backbone** | Parte da rede que extrai características da imagem. [doc 2](02-redes-neurais-e-cnns.md#backbone-neck-e-head) |
| **Batch** | Grupo de imagens processadas juntas em um passo de treino. |
| **Bounding box (bbox)** | Retângulo que envolve um objeto: `[x, y, largura, altura]`. |
| **Checkpoint** | Arquivo com os pesos de um modelo (`.pt`, `.pth`). |
| **Confiança (score)** | Número de 0 a 1 indicando o quão seguro o modelo está de uma previsão. |
| **CUDA** | Plataforma da NVIDIA que permite rodar cálculos na GPU. [doc 2](02-redes-neurais-e-cnns.md#gpu-e-cuda) |
| **Data augmentation** | Criar variações das imagens de treino (girar, espelhar, mudar brilho). [doc 5](05-datasets-e-anotacao.md#data-augmentation) |
| **Embedding** | Representação numérica compacta de algo (uma imagem, um texto). |
| **Endpoint** | Uma URL + método da API, ex.: `POST /detect`. [doc 8](08-arquitetura-da-aplicacao.md) |
| **Época (epoch)** | Uma passada completa por todas as imagens de treino. |
| **Fine-tuning** | Continuar o treino de um modelo pré-treinado com dados novos. [doc 2](02-redes-neurais-e-cnns.md#transfer-learning-e-fine-tuning) |
| **Foundation model** | Modelo enorme, treinado em muitos dados, que serve de base para várias tarefas (ex.: SAM). |
| **Gradiente** | Direção em que cada peso deve mudar para reduzir a loss. |
| **Hit-testing** | Descobrir qual elemento está sob o ponto clicado. [doc 7](07-mascaras-e-pos-processamento.md#ponto-em-polígono-hit-testing) |
| **Inferência** | Usar um modelo treinado para fazer previsões (sem treinar). |
| **IoU** | Intersection over Union — medida de sobreposição entre duas regiões. [doc 6](06-metricas.md#iou-intersection-over-union) |
| **Learning rate** | Tamanho do passo de ajuste dos pesos em cada iteração do treino. |
| **Loss** | Número que mede o quão erradas estão as previsões; o treino tenta minimizá-lo. |
| **mAP** | Mean Average Precision — métrica padrão de detectores. [doc 6](06-metricas.md#map-mean-average-precision) |
| **Máscara** | Array do tamanho da imagem marcando quais pixels pertencem a um objeto. [doc 7](07-mascaras-e-pos-processamento.md#máscara-binária) |
| **NMS** | Non-Maximum Suppression — remove previsões duplicadas do mesmo objeto. [doc 6](06-metricas.md#nms-non-maximum-suppression) |
| **Overfitting** | Modelo decora o treino e vai mal em imagens novas. [doc 5](05-datasets-e-anotacao.md#overfitting) |
| **Pesos** | Os parâmetros ajustáveis de uma rede neural; é o que o treino aprende. |
| **Polígono** | Lista de pontos `(x, y)` que descreve o contorno de um objeto. |
| **Precision / Recall** | "Quantos dos que marquei estão certos?" / "Quantos dos reais eu achei?" [doc 6](06-metricas.md#precision-recall-e-f1) |
| **Prompt** | Indicação dada ao SAM do que segmentar: ponto, caixa ou texto. [doc 3](03-segment-anything-sam.md#tipos-de-prompt) |
| **RLE** | Run-Length Encoding — forma compacta de guardar máscaras. |
| **Segmentação de instância** | Separar cada objeto individualmente, pixel a pixel. [doc 1](01-visao-computacional.md#as-principais-tarefas) |
| **Tensor** | Array multidimensional (como NumPy) que pode ficar na GPU. |
| **Transfer learning** | Reaproveitar o conhecimento de um modelo treinado em outra tarefa. |
| **VRAM** | Memória da GPU. Limita o tamanho do modelo e do batch. |
| **Zero-shot** | Executar uma tarefa sem ter sido treinado especificamente para ela. |

## Termos de escalada

| Termo | Significado |
|---|---|
| **Agarra (hold)** | Peça fixada na parede onde se apoiam mãos e pés. |
| **Volume** | Forma geométrica grande parafusada na parede; agarras podem ser fixadas nele. |
| **Jug** | Agarra grande e fácil de segurar ("balde"). |
| **Crimp** | Agarra pequena, de borda fina, segurada com a ponta dos dedos. |
| **Sloper** | Agarra arredondada, sem borda, que depende de atrito. |
| **Pinch** | Agarra segurada como uma pinça, com o polegar se opondo aos dedos. |
| **Pocket** | Agarra com um buraco onde entram um ou mais dedos. |
| **Boulder / via** | Sequência de agarras que forma um problema de escalada. |
