"""Interface web: anexe as imagens, revise o texto e salve.

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from guarani_ocr.pipeline import process_image, save_correction
from guarani_ocr.storage import Database, DriveSync
from guarani_ocr.training import export_ground_truth

st.set_page_config(page_title="OCR Guarani Antigo", layout="wide")


@st.cache_resource
def get_db():
    return Database()


@st.cache_resource
def get_drive():
    return DriveSync() if DriveSync.configured() else None


db = get_db()

with st.sidebar:
    st.header("Configuração")
    engines = (["tesseract"]
               + (["google"] if os.getenv("GOOGLE_VISION_API_KEY") else [])
               + (["claude"] if os.getenv("ANTHROPIC_API_KEY") else []))
    engine = st.radio("Motor de OCR", engines,
                      help="Tesseract: local e gratuito (treinável). "
                           "Google: Cloud Vision; exige GOOGLE_VISION_API_KEY. "
                           "Claude: exige ANTHROPIC_API_KEY.")
    book = st.text_input("Livro / obra", "Ara poru aguĳey haba")
    split = st.selectbox("Página dupla?", ["auto", "sim", "não"])
    use_drive = st.checkbox("Salvar no Google Drive", value=DriveSync.configured(),
                            disabled=not DriveSync.configured(),
                            help="Defina DRIVE_FOLDER_ID (e SHEET_ID) para ativar.")
    st.divider()
    if st.button("Exportar dados de treino"):
        s = export_ground_truth(db)
        st.success(f"{s['lines']} linhas de {s['pages']} páginas exportadas")
        for msg in s["skipped_pages"]:
            st.warning(msg)

drive = get_drive() if use_drive else None

tab_new, tab_review = st.tabs(["📤 Nova imagem", "📚 Páginas salvas"])

with tab_new:
    files = st.file_uploader("Anexe uma ou mais imagens",
                             type=["jpg", "jpeg", "png", "tif", "tiff"],
                             accept_multiple_files=True)
    if files and st.button("Extrair texto", type="primary"):
        split_opt = {"auto": None, "sim": True, "não": False}[split]
        for f in files:
            with st.spinner(f"Processando {f.name}…"):
                ids = process_image(f.getvalue(), f.name, db, engine, book,
                                    split_opt, drive)
            st.success(f"{f.name}: {len(ids)} página(s) salvas — revise na aba ao lado.")

with tab_review:
    pages = db.all()
    if not pages:
        st.info("Nenhuma página ainda.")
    else:
        labels = {
            p["id"]: f"{'✔ ' if p['corrected_text'] else ''}{p['source_name']} — "
                     f"página {p['page_index'] + 1}"
            for p in pages
        }
        page_id = st.selectbox("Página", list(labels), format_func=labels.get,
                               index=len(pages) - 1)
        p = db.get(page_id)
        left, right = st.columns(2)
        with left:
            st.image(p["image_path"], use_container_width=True)
            info = p["engine"] or ""
            if p["confidence"] is not None:
                info += f" · confiança média {p['confidence']:.0f}%"
            st.caption(info)
        with right:
            text = st.text_area(
                "Transcrição (mantenha uma linha de texto por linha impressa)",
                p["corrected_text"] or p["raw_text"] or "", height=600,
                key=f"txt{page_id}")
            st.caption("Copiar: ſ  ã ẽ ĩ õ ũ ỹ  g̃  à è ì ò ù  â ê î ô û ŷ  ä ë ï ö ü ÿ  ç ñ  &")
            if st.button("Salvar correção", type="primary"):
                save_correction(db, page_id, text, drive)
                st.success("Salvo!")
            with st.expander("Versão modernizada (ſ→s, hifenização juntada)"):
                st.write(db.get(page_id)["modern_text"])
