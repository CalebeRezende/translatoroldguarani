"""Linha de comando.

  python -m guarani_ocr.cli ocr pasta_ou_imagem [--engine claude] [--book "Ara poru"] [--drive]
  python -m guarani_ocr.cli correct ID arquivo.txt [--drive]
  python -m guarani_ocr.cli list
  python -m guarani_ocr.cli export-gt
  python -m guarani_ocr.cli export-txt saida/
  python -m guarani_ocr.cli eval [--lang grn_old]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import iter_images, process_image, save_correction
from .storage import Database, DriveSync
from .training import export_ground_truth


def main(argv=None):
    ap = argparse.ArgumentParser(prog="guarani_ocr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("ocr", help="faz OCR de uma imagem ou pasta")
    o.add_argument("path", type=Path)
    o.add_argument("--engine", default="tesseract", choices=["tesseract", "claude"])
    o.add_argument("--book")
    o.add_argument("--split", choices=["auto", "yes", "no"], default="auto",
                   help="dividir página dupla")
    o.add_argument("--drive", action="store_true", help="enviar ao Google Drive")

    c = sub.add_parser("correct", help="salva a transcrição corrigida de uma página")
    c.add_argument("id", type=int)
    c.add_argument("file", type=Path)
    c.add_argument("--drive", action="store_true")

    sub.add_parser("list", help="lista as páginas no banco")
    sub.add_parser("export-gt", help="gera ground truth para treinar o Tesseract")
    e = sub.add_parser("eval", help="mede a taxa de erro (CER) nas páginas corrigidas")
    e.add_argument("--lang", help="modelo do Tesseract (padrão: OCR_TESS_LANG ou spa_old)")
    t = sub.add_parser("export-txt", help="exporta todos os textos para uma pasta")
    t.add_argument("out", type=Path)

    a = ap.parse_args(argv)
    db = Database()
    drive = DriveSync() if getattr(a, "drive", False) else None

    if a.cmd == "ocr":
        split = {"auto": None, "yes": True, "no": False}[a.split]
        files = iter_images(a.path) if a.path.is_dir() else [a.path]
        for f in files:
            ids = process_image(f, f.name, db, a.engine, a.book, split, drive)
            for i in ids:
                p = db.get(i)
                conf = f" conf={p['confidence']:.0f}" if p["confidence"] else ""
                print(f"[{i}] {f.name} p{p['page_index']}{conf}")
    elif a.cmd == "correct":
        save_correction(db, a.id, a.file.read_text(encoding="utf-8"), drive)
        print("ok")
    elif a.cmd == "list":
        for p in db.all():
            mark = "✔" if p["corrected_text"] else " "
            print(f"{mark} [{p['id']}] {p['book'] or '-'} | {p['source_name']} "
                  f"p{p['page_index']} | {p['engine']}")
    elif a.cmd == "export-gt":
        s = export_ground_truth(db)
        print(f"{s['lines']} linhas de {s['pages']} páginas exportadas")
        for msg in s["skipped_pages"]:
            print("  pulada:", msg)
    elif a.cmd == "eval":
        from .evaluate import evaluate
        r = evaluate(db, a.lang)
        for name, cer in r["pages"]:
            print(f"  {name}: CER {cer:.1%}")
        print(f"CER médio ({r['lang']}): {r['cer']:.1%} em {len(r['pages'])} páginas")
    elif a.cmd == "export-txt":
        a.out.mkdir(parents=True, exist_ok=True)
        for p in db.all():
            stem = f"{Path(p['source_name']).stem}_p{p['page_index']}"
            (a.out / f"{stem}.txt").write_text(
                p["corrected_text"] or p["raw_text"] or "", encoding="utf-8")
        print(f"textos salvos em {a.out}")


if __name__ == "__main__":
    main()
