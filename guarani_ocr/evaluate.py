"""Mede a qualidade do OCR: CER (taxa de erro por caractere) contra o texto corrigido.

Para uma medida honesta, avalie em páginas que NÃO foram usadas no treino.
"""
from __future__ import annotations

import cv2

from . import postprocess
from .engines import TesseractEngine, get_engine
from .storage import Database, page_name


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(text: str) -> str:
    # compara só o conteúdo: ignora itálico, marcas de dúvida e espaço/quebra de linha
    text = text.replace("_", "").replace("[?]", "").replace("[ilegível]", "")
    return " ".join(text.split())


def evaluate(db: Database, lang: str | None = None,
             only: set[str] | None = None, engine: str = "tesseract") -> dict:
    eng = TesseractEngine(lang=lang) if engine == "tesseract" else get_engine(engine)
    pages, errs, total = [], 0, 0
    for p in db.corrected():
        if only and page_name(p) not in only:
            continue
        img = cv2.imread(p["image_path"], cv2.IMREAD_GRAYSCALE)
        hyp = _norm(postprocess.clean(eng.recognize(img).text))
        ref = _norm(p["corrected_text"])
        d = levenshtein(hyp, ref)
        errs, total = errs + d, total + len(ref)
        pages.append((page_name(p), d / max(len(ref), 1)))
    return {"lang": getattr(eng, "lang", engine), "pages": pages, "cer": errs / max(total, 1)}
