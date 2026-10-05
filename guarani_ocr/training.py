"""Exporta páginas corrigidas como ground truth linha a linha para o tesstrain.

Cada linha vira um par  <nome>.png + <nome>.gt.txt  em training/ground-truth/.
Cada linha detectada pelo Tesseract é pareada com a linha corrigida mais
parecida (em ordem), então linhas extras ou faltantes não estragam o
alinhamento. Ainda assim, ao corrigir, mantenha uma linha de texto por linha
impressa. Linhas com [ilegível] ou [?] não entram no treino.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2

from .engines import TesseractEngine
from .storage import Database, page_name

SKIP = re.compile(r"\[(ilegível|\?)\]")


def _gt_line(line: str) -> str:
    return line.replace("_", "").strip()  # remove a marcação de itálico


def _drop_capital(ocr: str, gt: str) -> str:
    """A capitular ornamentada (o "P" grande de "POR") fica fora da caixa da
    linha; se o OCR leu "OR orden", o gabarito do recorte também deve ser "OR"."""
    if len(gt) > 1 and gt[0].isupper() and gt[1].isupper():
        if SequenceMatcher(None, ocr, gt[1:]).ratio() > SequenceMatcher(None, ocr, gt).ratio():
            return gt[1:]
    return gt


def align(boxes: list[dict], gt: list[str], min_ratio: float = 0.5,
          window: int = 3) -> list[tuple[dict, str]]:
    """Pareia caixas de linha (com o texto do OCR) às linhas corrigidas."""
    pairs, j = [], 0
    for box in boxes:
        best, best_k = 0.0, None
        for k in range(j, min(j + window, len(gt))):
            r = SequenceMatcher(None, box["text"], gt[k]).ratio()
            if r > best:
                best, best_k = r, k
        if best_k is not None and best >= min_ratio:
            pairs.append((box, _drop_capital(box["text"], gt[best_k])))
            j = best_k + 1
    return pairs


def export_ground_truth(db: Database, out_dir: Path | str = "training/ground-truth",
                        pad: int = 6, exclude: set[str] = frozenset()) -> dict:
    """`exclude`: nomes de páginas (ex.: "prologo_p1") reservadas para teste."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg = TesseractEngine()
    stats = {"pages": 0, "lines": 0, "skipped_pages": []}
    for p in db.corrected():
        if page_name(p) in exclude:
            continue
        img = cv2.imread(p["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        boxes = json.loads(p["lines_json"] or "[]")
        if not boxes:
            boxes = seg.recognize(img).lines
        gt = [l.strip() for l in p["corrected_text"].splitlines() if l.strip()]
        pairs = align(boxes, gt)
        if len(pairs) < len(gt) * 0.5:
            stats["skipped_pages"].append(
                f"{p['source_name']} p{p['page_index']}: só {len(pairs)} de {len(gt)} "
                "linhas alinhadas")
            continue
        stats["pages"] += 1
        stem = page_name(p)
        h, w = img.shape
        for n, (ln, text) in enumerate(pairs):
            if SKIP.search(text):
                continue
            x0, y0, x1, y1 = ln["box"]
            crop = img[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]
            name = out / f"{stem}_l{n:03d}"
            cv2.imwrite(f"{name}.png", crop)
            Path(f"{name}.gt.txt").write_text(_gt_line(text) + "\n", encoding="utf-8")
            stats["lines"] += 1
    return stats
