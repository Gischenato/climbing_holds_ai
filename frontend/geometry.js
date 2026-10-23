// Geometria pura (sem DOM): fácil de testar fora do navegador.

/** Área de um polígono [[x, y], ...] pela fórmula do cadarço (shoelace). */
export function polygonArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

/** Ponto dentro do polígono? Ray casting: raio para a direita, conta cruzamentos (ímpar = dentro). */
export function pointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * Agarra clicada em (x, y), em coordenadas da imagem.
 * Se o ponto cair em várias (sobrepostas), devolve a de menor área — normalmente a
 * agarra pequena em cima de uma grande. `holds` precisa ter `polygon` e `area`.
 */
export function pickHold(holds, x, y) {
  let best = null;
  for (const hold of holds) {
    const [bx, by, bw, bh] = hold.bbox;
    if (x < bx || y < by || x > bx + bw || y > by + bh) continue; // descarte rápido pela caixa
    if (!pointInPolygon(x, y, hold.polygon)) continue;
    if (best === null || hold.area < best.area) best = hold;
  }
  return best;
}

/**
 * Novo estado de visualização {scale, x, y} após um zoom de `factor` ancorado no ponto (px, py)
 * da moldura (vw × vh). O ponto da imagem sob o cursor fica parado; a imagem sempre cobre a
 * moldura e o zoom fica entre 1 (cabe na tela) e `maxScale`.
 */
export function zoomAt(view, px, py, factor, vw, vh, maxScale = 20) {
  const scale = Math.min(maxScale, Math.max(1, view.scale * factor));
  const k = scale / view.scale;
  return clampView({ scale, x: px - (px - view.x) * k, y: py - (py - view.y) * k }, vw, vh);
}

/** Mantém a imagem ampliada cobrindo toda a moldura (sem bordas vazias). */
export function clampView(view, vw, vh) {
  return {
    scale: view.scale,
    x: Math.min(0, Math.max(view.x, vw * (1 - view.scale))),
    y: Math.min(0, Math.max(view.y, vh * (1 - view.scale))),
  };
}
