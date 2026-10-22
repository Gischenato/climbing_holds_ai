# 8. Arquitetura da aplicação

## O que é

A aplicação tem duas partes que conversam por HTTP:

- **Backend** (servidor, Python): recebe a imagem, roda o modelo na GPU, devolve as agarras.
- **Frontend** (cliente, navegador): mostra a imagem, desenha as agarras e trata os cliques.

```
 Navegador                                   Servidor (FastAPI)
 ─────────                                   ──────────────────
 1. usuário escolhe a foto
 2. POST /detect  (multipart, imagem) ─────▶ 3. lê a imagem
                                             4. detector.detect(img)  ← GPU
                                             5. filtra + polígonos
 7. desenha polígonos no canvas  ◀───────── 6. JSON { holds: [...] }
 8. clique → seleciona/deseleciona
    (tudo local, sem falar com o servidor)
```

Repare que **selecionar/deselecionar não precisa do backend**: depois que o frontend recebeu os polígonos, o clique é resolvido no navegador, instantâneo.

### REST e JSON

**REST** é um estilo de API sobre HTTP: cada recurso tem uma URL e usamos métodos (`GET`, `POST`, …). **JSON** é o formato de texto usado na resposta.

Nossa API (por enquanto) tem um endpoint:

```
POST /detect
Content-Type: multipart/form-data
Body: file=<imagem>

200 OK
{
  "width": 4000,
  "height": 3000,
  "holds": [
    { "id": 0, "polygon": [[1650, 900], [1700, 890], ...], "bbox": [1640, 885, 80, 70], "score": 0.93 }
  ]
}
```

As coordenadas estão em **pixels da imagem original** — o frontend converte para a tela.

### FastAPI

**FastAPI** é um framework web Python moderno:

- Tipagem com type hints → validação automática dos dados.
- Gera documentação interativa em `/docs` (Swagger) para testar a API pelo navegador.
- Suporta `async` e upload de arquivos com `UploadFile`.

### Multipart/form-data

É o formato HTTP usado para **enviar arquivos** (o mesmo de um `<form>` com `<input type="file">`). O JSON não carrega bem arquivos binários; o multipart sim.

### Carregar o modelo uma vez

Carregar o SAM do disco para a GPU leva vários segundos e ocupa gigabytes de VRAM. Se fizermos isso **a cada requisição**, cada detecção fica lenta e a memória pode estourar.

Solução: carregar **no startup** do servidor (com o `lifespan` do FastAPI) e reutilizar o mesmo objeto em todas as requisições.

### A interface `Detector`

Para trocar SAM por YOLO sem mudar a API nem o frontend:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Hold:
    polygon: list[list[int]]
    bbox: list[int]
    score: float

class Detector(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[Hold]: ...

class SamDetector(Detector): ...   # Fase 1/2
class YoloDetector(Detector): ...  # Fase 5
```

Isso é o **princípio de inversão de dependência**: a API depende da *abstração* (`Detector`), não de um modelo específico.

## Coordenadas imagem ↔ tela

A foto tem 4000×3000 px, mas aparece na tela, por exemplo, com 1000×750. É preciso converter:

```js
const scale = canvas.width / image.naturalWidth;   // ex.: 1000 / 4000 = 0.25

// imagem → tela (para desenhar)
const xTela = xImg * scale;

// tela → imagem (para o clique)
const rect = canvas.getBoundingClientRect();
const xImg = (event.clientX - rect.left) / scale;
const yImg = (event.clientY - rect.top)  / scale;
```

Uma alternativa mais simples: aplicar `ctx.scale(scale, scale)` e desenhar tudo em coordenadas da imagem.

> Cuidado com **EXIF orientation**: fotos de celular às vezes estão "deitadas" e só aparecem em pé por causa de um metadado. Se o backend e o navegador interpretarem diferente, os polígonos ficam desalinhados. Corrija no backend com `PIL.ImageOps.exif_transpose`.

## Exemplo

Esqueleto do backend:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles

detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = SamDetector("models/sam_checkpoint.pt")   # carrega 1x
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/detect")
async def detect(file: UploadFile):
    image = ler_imagem_rgb(await file.read())
    holds = detector.detect(image)
    return {"width": image.shape[1], "height": image.shape[0],
            "holds": [{"id": i, **vars(h)} for i, h in enumerate(holds)]}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

Rodar: `uvicorn backend.main:app --reload` e abrir `http://localhost:8000`.

## Para ir além

- [FastAPI — Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI — Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [MDN — Canvas API Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial)
- [MDN — Using Fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)
