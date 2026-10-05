"""Normalização do texto: versão diplomática (fiel) e versão modernizada."""
from __future__ import annotations

import re
import unicodedata

# Trocas da grafia tipográfica antiga para a moderna (só na versão "modernizada").
MODERN_MAP = {
    "ſ": "s",
    "ꝑ": "per",
    "ꝓ": "pro",
    "&c.": "etc.",
}


def clean(text: str) -> str:
    """Limpeza segura, aplicada sempre: NFC, espaços e lixo de borda."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # hífen invisível
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        # linhas de 1–2 caracteres sem letra costumam ser sujeira da digitalização
        if line and len(line) <= 2 and not re.search(r"\w", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def join_hyphenation(text: str) -> str:
    """Junta palavras divididas no fim da linha (Com-\\npañia -> Compañia)."""
    return re.sub(r"(\w)[-¬=]\n\s*(\w)", r"\1\2", text)


def modernize(text: str) -> str:
    """Texto corrido para leitura/tradução: ſ->s, junta hifenização e parágrafos."""
    for old, new in MODERN_MAP.items():
        text = text.replace(old, new)
    text = join_hyphenation(text)
    # quebras simples viram espaço; linhas em branco separam parágrafos
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)
