# Referência da API

Documentação de como a API do detector de agarras funciona. Código em [backend/main.py](../backend/main.py).
Para o conceito por trás (REST, multipart, FastAPI), veja [08 – Arquitetura da aplicação](08-arquitetura-da-aplicacao.md).

> **Manter atualizado:** qualquer mudança em endpoints, campos da resposta, limites ou erros em `backend/` deve ser refletida neste arquivo na mesma alteração.

## Como rodar

A partir da raiz do projeto, com o `.venv` ativo:

```
uvicorn backend.main:app --reload
```

O modelo (SAM 2.1, ~900 MB) é carregado na subida; o servidor só atende depois de "Application startup complete". Endereço padrão: `http://localhost:8000`.

| URL | O que é |
|---|---|
| `http://localhost:8000/` | O frontend (pasta `frontend/`). Enquanto ela estiver vazia, devolve `{"detail":"Not Found"}` |
| `http://localhost:8000/docs` | Swagger: testa a API pelo navegador |
| `http://localhost:8000/openapi.json` | Descrição da API em formato OpenAPI |

## Endpoints

### `POST /detect`

Recebe uma foto e devolve as agarras encontradas.

**Requisição:** `multipart/form-data` com um campo `file` contendo a imagem.

| Item | Regra |
|---|---|
| Campo | `file` (obrigatório) |
| Formatos | Qualquer um que o Pillow abra (JPEG, PNG, WebP…) |
| Tamanho máximo | 20 MB |
| Orientação | A orientação EXIF (fotos de celular) é aplicada pelo servidor |
| Tamanho da imagem | Se o lado maior passar de 2048 px, a imagem é reduzida mantendo a proporção. `width`/`height` da resposta refletem o tamanho **depois** disso |

**Resposta `200 OK`:**

```json
{
  "width": 480,
  "height": 640,
  "holds": [
    {
      "id": 0,
      "polygon": [[394, 69], [391, 69], [389, 71], [388, 78]],
      "bbox": [349, 62, 48, 40],
      "score": 0.93
    }
  ]
}
```

| Campo | Tipo | Significado |
|---|---|---|
| `width`, `height` | inteiro | Tamanho da imagem processada, em pixels. Todas as coordenadas abaixo estão neste sistema |
| `holds` | lista | Agarras encontradas. Pode ser vazia |
| `holds[].id` | inteiro | Índice da agarra (0, 1, 2…). **Vale só dentro desta resposta**: a mesma foto enviada de novo pode gerar ids diferentes |
| `holds[].polygon` | lista de `[x, y]` | Contorno externo da agarra, com pelo menos 3 pontos. `x` = distância da borda esquerda, `y` = da borda de cima, em pixels. Simplificado (não é pixel a pixel) e sem buracos internos |
| `holds[].bbox` | `[x, y, largura, altura]` | Retângulo que envolve a agarra; `x`,`y` é o canto superior esquerdo |
| `holds[].score` | número de 0 a 1 | Nota que o SAM dá à qualidade do recorte (`predicted_iou`). **Não** é a certeza de ser uma agarra |

Origem das coordenadas: canto superior esquerdo da imagem; `y` cresce para baixo.

**Erros:**

| Código | Quando | Corpo |
|---|---|---|
| `400` | O arquivo não é uma imagem válida | `{"detail": "Arquivo não é uma imagem válida"}` |
| `413` | O arquivo passa de 20 MB | `{"detail": "Imagem maior que 20 MB"}` |
| `422` | O campo `file` não foi enviado | Corpo de validação padrão do FastAPI |

**Desempenho e concorrência:**
- ~7 s por foto em uma RTX 4070 SUPER (com 64 pontos por lado no SAM).
- O endpoint é síncrono e usa um *lock*: só **uma** detecção roda na GPU por vez; requisições simultâneas esperam na fila.
- Não há cache: enviar a mesma foto duas vezes roda o modelo duas vezes.

## Exemplos de uso

**curl**

```
curl -F "file=@minha_parede.jpg" http://localhost:8000/detect
```

**JavaScript (navegador)**

```js
const form = new FormData();
form.append("file", inputFile.files[0]);
const resp = await fetch("/detect", { method: "POST", body: form });
if (!resp.ok) throw new Error((await resp.json()).detail);
const { width, height, holds } = await resp.json();
```

**Python**

```python
import requests

with open("minha_parede.jpg", "rb") as f:
    data = requests.post("http://localhost:8000/detect", files={"file": f}).json()
print(len(data["holds"]), "agarras")
```

## Como a resposta é produzida

```
upload ──▶ read_image ──▶ detector.detect ──▶ JSON
          (EXIF, RGB,    SAM 2.1 → filtros → polígonos
           redução)       (backend/detectors.py)
```

1. `read_image` ([main.py](../backend/main.py)) abre a imagem, aplica a orientação EXIF, converte para RGB e reduz se passar de 2048 px.
2. `SamDetector.detect` ([detectors.py](../backend/detectors.py)) gera as máscaras com o SAM 2.1 e filtra o que não parece agarra ([holds.py](../backend/holds.py)): área mínima (0,01% da imagem) e máxima (5%), formato muito alongado, máscaras grandes tocando a borda e duplicatas (IoU > 0,5).
3. Cada máscara restante vira um polígono (`cv2.findContours` + `approxPolyDP`).

Os parâmetros do SAM e dos filtros foram escolhidos medindo contra o dataset: recall ≈ 80%, precision ≈ 66%. Ou seja, algumas agarras podem faltar e algumas marcações não são agarras (painéis, discos no chão). Detalhes no [TODO](../TODO.md).

## Trocar o modelo

A API depende só da interface `Detector` ([detectors.py](../backend/detectors.py)): `detect(image_rgb) -> list[Hold]`. Para usar o YOLO na Fase 5, basta criar um `YoloDetector` e instanciá-lo no `lifespan` em [main.py](../backend/main.py); o formato da resposta não muda.

## Histórico

| Data | Mudança |
|---|---|
| 2026-10-22 | Criação: `POST /detect` com SAM 2.1; frontend servido em `/` |
