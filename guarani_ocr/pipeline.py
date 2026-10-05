"""Junta tudo: imagem -> páginas -> OCR -> banco (-> Drive)."""
from __future__ import annotations

from pathlib import Path

import cv2

from . import postprocess, preprocess
from .engines import get_engine
from .storage import Database, DriveSync


def process_image(src, source_name: str, db: Database, engine: str = "tesseract",
                  book: str | None = None, split: bool | None = None,
                  drive: DriveSync | None = None) -> list[int]:
    """Processa uma imagem (caminho ou bytes) e salva cada página. Retorna os ids."""
    ocr = get_engine(engine)
    gray = preprocess.load(src)
    ids = []
    for i, page in enumerate(preprocess.split_spread(gray, split)):
        page = preprocess.deskew(page)
        bw = preprocess.crop_margins(preprocess.binarize(page))
        img_path = db.image_path_for(source_name, i)
        cv2.imwrite(str(img_path), bw)
        # Tesseract lê melhor a imagem binarizada; Claude, a imagem em cinza.
        result = ocr.recognize(bw if engine == "tesseract" else page)
        text = postprocess.clean(result.text)
        page_id = db.upsert(
            source_name=source_name, page_index=i, book=book,
            engine=result.engine, confidence=result.confidence,
            raw_text=text, modern_text=postprocess.modernize(text),
            lines=result.lines, image_path=str(img_path),
        )
        if drive:
            drive.push(db, page_id)
        ids.append(page_id)
    return ids


def save_correction(db: Database, page_id: int, text: str,
                    drive: DriveSync | None = None) -> None:
    p = db.get(page_id)
    text = postprocess.clean(text)
    db.upsert(source_name=p["source_name"], page_index=p["page_index"],
              corrected_text=text, modern_text=postprocess.modernize(text))
    if drive:
        drive.push(db, page_id)


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
