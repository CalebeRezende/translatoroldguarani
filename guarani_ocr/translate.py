"""Tradução do guarani antigo (Qwen ou Claude) a partir da versão modernizada."""
from __future__ import annotations

import os

from .engines import qwen_api_key, qwen_chat

TRANSLATE_PROMPT = """Você é especialista em guarani jesuítico (séculos XVII–XVIII, \
grafia das Missões do Paraguai: ſ, ĭ, ŷ, g̃, marcas nasais com circunflexo).

Traduza o texto abaixo para {lang}. O texto saiu de um OCR e pode ter erros: \
interprete pelo contexto. Trechos em espanhol ou latim também vão traduzidos. \
Termos religiosos: Tupã/Tûpâ = Deus, Tupã Ra'y = Filho de Deus, Ñandeyara/Cheyara = \
Nosso/Meu Senhor, Santa María, ângaipa = pecado, Sacramento etc. Mantenha nomes \
próprios. Onde a leitura for incerta, ponha [?] após a palavra traduzida.

Responda SOMENTE com a tradução, em parágrafos, sem comentários.

Texto:
{text}"""

LANGUAGES = ["português", "espanhol", "inglês"]


def available_translators() -> list[str]:
    return ((["qwen"] if qwen_api_key() else [])
            + (["claude"] if os.getenv("ANTHROPIC_API_KEY") else []))


def translate(text: str, provider: str = "qwen", lang: str = "português") -> str:
    if not text.strip():
        return ""
    prompt = TRANSLATE_PROMPT.format(lang=lang, text=text)
    if provider == "qwen":
        model = os.getenv("TRANSLATE_QWEN_MODEL", "qwen-max")
        return qwen_chat(prompt, model).strip()
    if provider == "claude":
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=os.getenv("OCR_CLAUDE_MODEL", "claude-opus-5-5"), max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
    raise ValueError(f"Tradutor desconhecido: {provider}")
