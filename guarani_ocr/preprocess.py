"""Pré-processamento: divide páginas duplas, corrige inclinação e binariza."""
from __future__ import annotations

import cv2
import numpy as np


def load(path_or_bytes) -> np.ndarray:
    """Carrega uma imagem (caminho ou bytes) em tons de cinza."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Não foi possível ler a imagem")
    return img


def split_spread(gray: np.ndarray, force: bool | None = None) -> list[np.ndarray]:
    """Divide uma digitalização de página dupla na dobra central.

    `force=None` decide sozinho (imagem mais larga que alta); True/False força.
    A dobra é a coluna, no terço central, com menos tinta.
    """
    h, w = gray.shape
    is_spread = w > h * 1.2 if force is None else force
    if not is_spread:
        return [gray]
    ink = (255 - gray).astype(np.float32)
    lo, hi = int(w * 0.35), int(w * 0.65)
    profile = ink[:, lo:hi].sum(axis=0)
    profile = np.convolve(profile, np.ones(25) / 25, mode="same")
    cut = lo + int(np.argmin(profile))
    return [gray[:, :cut], gray[:, cut:]]


def deskew(gray: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Corrige pequena inclinação procurando o ângulo que mais "afina" as linhas."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5)
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    best, best_score = 0.0, -1.0
    h, w = bw.shape
    for angle in np.arange(-max_angle, max_angle + 0.01, 0.25):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = angle, score
    if abs(best) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Remove ruído leve e binariza (Otsu) — bom para impressão antiga contrastada."""
    den = cv2.fastNlMeansDenoising(gray, None, h=10)
    return cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def crop_margins(bw: np.ndarray, pad: int = 20) -> np.ndarray:
    """Recorta as margens vazias e as bordas escuras da digitalização."""
    h, w = bw.shape
    # descarta 2% de cada borda, onde costuma haver sombra da encadernação
    mx, my = int(w * 0.02), int(h * 0.02)
    inner = bw[my:h - my, mx:w - mx]
    ink = 255 - inner
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return bw
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, inner.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, inner.shape[0])
    return inner[y0:y1, x0:x1]


def prepare(path_or_bytes, split: bool | None = None) -> list[np.ndarray]:
    """Pipeline completo: retorna uma imagem binarizada por página."""
    gray = load(path_or_bytes)
    pages = []
    for page in split_spread(gray, split):
        page = deskew(page)
        pages.append(crop_margins(binarize(page)))
    return pages


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Falha ao codificar PNG")
    return buf.tobytes()
