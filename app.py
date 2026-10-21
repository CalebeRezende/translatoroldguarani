"""Interface web: anexe as imagens, revise o texto e a tradução, e salve na planilha.

    streamlit run app.py
"""
from __future__ import annotations

import hmac
import json
import os

import streamlit as st

from guarani_ocr import postprocess
from guarani_ocr.engines import available_engines
from guarani_ocr.pipeline import process_image, save_correction
from guarani_ocr.storage import Database, DriveSync
from guarani_ocr.training import export_ground_truth
from guarani_ocr.translate import LANGUAGES, available_translators, translate

st.set_page_config(page_title="OCR Guarani Antigo", layout="wide")


def load_secrets() -> None:
    """No Streamlit Cloud as chaves ficam em "Secrets" (TOML); vira variável de
    ambiente, como na execução local. A tabela [gcp_service_account] vira JSON."""
    try:
        secrets = st.secrets.to_dict()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return
    for k, v in secrets.items():
        if k == "gcp_service_account":
            os.environ.setdefault("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(v))
        elif isinstance(v, (str, int, float)):
            os.environ.setdefault(k, str(v))


def require_password() -> None:
    """Se APP_PASSWORD estiver definida, pede a senha (protege a sua chave da API)."""
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return
    pw = st.text_input("Senha", type="password")
    if pw and hmac.compare_digest(pw, expected):
        st.session_state["auth_ok"] = True
        st.rerun()
    if pw:
        st.error("Senha incorreta")
    st.stop()


load_secrets()
require_password()


@st.cache_resource
def get_db():
    return Database()


@st.cache_resource
def get_drive():
    return DriveSync() if DriveSync.configured() else None


db = get_db()

with st.sidebar:
    st.header("Configuração")
    engine = st.radio("Motor de OCR", available_engines(),
                      help="Tesseract: local e gratuito (treinável). "
                           "Qwen: exige DASHSCOPE_API_KEY. "
                           "Google: Cloud Vision; exige GOOGLE_VISION_API_KEY. "
                           "Claude: exige ANTHROPIC_API_KEY.")
    translators = available_translators()
    translator = st.radio("Tradutor", translators or ["(nenhum)"],
                          disabled=not translators,
                          help="Defina DASHSCOPE_API_KEY (Qwen) ou ANTHROPIC_API_KEY.")
    target_lang = st.selectbox("Traduzir para", LANGUAGES)
    book = st.text_input("Livro / obra", "Ara poru aguĳey haba")
    split = st.selectbox("Página dupla?", ["auto", "sim", "não"])
    if not DriveSync.configured():
        st.info("Planilha Google desativada: defina SHEET_ID (e, se quiser, "
                "DRIVE_FOLDER_ID). Os textos ficam salvos no banco local.")
    st.divider()
    if st.button("Exportar dados de treino"):
        s = export_ground_truth(db)
        st.success(f"{s['lines']} linhas de {s['pages']} páginas exportadas")
        for msg in s["skipped_pages"]:
            st.warning(msg)


tab_new, tab_review = st.tabs(["📤 Nova imagem", "📚 Páginas salvas"])

with tab_new:
    files = st.file_uploader("Anexe uma ou mais imagens",
                             type=["jpg", "jpeg", "png", "tif", "tiff"],
                             accept_multiple_files=True)
    if files and st.button("Extrair texto", type="primary"):
        split_opt = {"auto": None, "sim": True, "não": False}[split]
        for f in files:
            with st.spinner(f"Processando {f.name}…"):
                # nada vai à planilha ainda: primeiro você revisa na outra aba
                ids = process_image(f.getvalue(), f.name, db, engine, book, split_opt)
            st.session_state["last_page"] = ids[0]
            st.success(f"{f.name}: {len(ids)} página(s) lidas — revise e traduza "
                       "na aba 📚 antes de enviar à planilha.")

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
        ids = list(labels)
        last = st.session_state.get("last_page")
        page_id = st.selectbox("Página", ids, format_func=labels.get,
                               index=ids.index(last) if last in ids else len(ids) - 1)
        p = db.get(page_id)
        left, right = st.columns(2)
        with left:
            st.image(p["image_path"], width="stretch")
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

        tr_key = f"tr{page_id}"
        st.session_state.setdefault(tr_key, p["translation"] or "")
        st.subheader(f"Tradução ({target_lang})")
        if st.button("🌐 Traduzir texto acima", disabled=not translators,
                     help="Usa o texto que está na caixa, já com as suas edições."):
            with st.spinner("Traduzindo…"):
                try:
                    st.session_state[tr_key] = translate(
                        postprocess.modernize(postprocess.clean(text)),
                        translator, target_lang)
                except Exception as e:  # erro da API: mostra e deixa editar à mão
                    st.error(str(e))
        translation = st.text_area("Revise a tradução antes de salvar",
                                   height=300, key=tr_key)

        c1, c2 = st.columns(2)
        if c1.button("💾 Salvar (só no banco local)"):
            save_correction(db, page_id, text, translation=translation)
            st.success("Salvo no banco local.")
        if c2.button("📤 Salvar e enviar à planilha", type="primary",
                     disabled=not DriveSync.configured()):
            with st.spinner("Enviando à Planilha Google…"):
                save_correction(db, page_id, text, get_drive(), translation)
            st.success("Salvo e enviado à planilha!")
        with st.expander("Versão modernizada (ſ→s, hifenização juntada)"):
            st.write(db.get(page_id)["modern_text"])
