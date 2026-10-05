"""Motores de OCR: Tesseract (local, treinável) e Claude (visão, alta precisão)."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field

import numpy as np
import pytesseract

from .preprocess import to_png_bytes


@dataclass
class OcrResult:
    text: str
    engine: str
    confidence: float | None = None  # média 0–100 (só Tesseract)
    lines: list[dict] = field(default_factory=list)  # caixas das linhas (Tesseract)


class TesseractEngine:
    """Tesseract LSTM. Use `lang="spa_old"` (espanhol antigo, com ſ) até treinar
    o seu próprio modelo, ex.: `lang="grn_old"` (ver training/README.md)."""

    name = "tesseract"

    def __init__(self, lang: str | None = None, psm: int = 4):
        self.lang = lang or os.getenv("OCR_TESS_LANG", "spa_old")
        # psm 4 = uma coluna de texto com linhas de tamanhos variados
        self.config = f"--psm {psm} -c preserve_interword_spaces=1"

    def recognize(self, img: np.ndarray) -> OcrResult:
        data = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        lines: dict[tuple, dict] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            ln = lines.setdefault(key, {"words": [], "box": [x, y, x + w, y + h]})
            ln["words"].append(word)
            b = ln["box"]
            ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
            if float(data["conf"][i]) >= 0:
                confs.append(float(data["conf"][i]))
        ordered = [
            {"text": " ".join(v["words"]), "box": v["box"]}
            for _, v in sorted(lines.items())
        ]
        text = "\n".join(l["text"] for l in ordered)
        conf = sum(confs) / len(confs) if confs else None
        return OcrResult(text=text, engine=f"tesseract:{self.lang}",
                         confidence=conf, lines=ordered)


CLAUDE_PROMPT = """Você é um paleógrafo especialista em impressos jesuíticos das \
Missões do Paraguai (séculos XVII–XVIII), em guarani antigo, espanhol e latim.

Transcreva o texto desta página de forma DIPLOMÁTICA (fiel à fonte):
- Mantenha a grafia original: ſ (s longo), &, abreviaturas (Doct., S. M., &c.), \
acentos graves (à, è, ò, vì) e circunflexos tal como impressos.
- Em guarani preserve os diacríticos nasais e guturais: ã ẽ ĩ õ ũ ỹ, g̃, ĝ, \
ÿ, î, û, ŷ, ĭ etc. Não "corrija" para o guarani moderno.
- Mantenha exatamente as quebras de linha da página e a hifenização no fim da linha.
- Inclua cabeçalho, título corrente, número/assinatura da página e reclamo \
(a palavra solta no pé da página), cada um em sua própria linha.
- Itálicos: envolva com _sublinhados_.
- Letra capitular: junte-a à palavra (ex.: "POR orden").
- Se algo estiver ilegível, escreva [ilegível]; se estiver em dúvida, \
coloque a leitura provável seguida de [?].

Responda SOMENTE com a transcrição, sem comentários."""


class ClaudeEngine:
    """OCR com Claude (visão). Requer ANTHROPIC_API_KEY. Excelente em tipos
    antigos e diacríticos raros; recebe a imagem em tons de cinza (não binarizada)."""

    name = "claude"

    def __init__(self, model: str | None = None, effort: str = "medium",
                 extra_instructions: str = ""):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("OCR_CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort
        self.prompt = CLAUDE_PROMPT + ("\n\n" + extra_instructions if extra_instructions else "")

    def recognize(self, img: np.ndarray) -> OcrResult:
        data = base64.standard_b64encode(to_png_bytes(img)).decode()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": self.effort},
            # Se o filtro de segurança recusar por engano, a API tenta outro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude recusou a transcrição desta página")
        text = "".join(b.text for b in response.content if b.type == "text")
        return OcrResult(text=text.strip(), engine=f"claude:{response.model}")


def get_engine(name: str, **kwargs):
    if name == "tesseract":
        return TesseractEngine(**kwargs)
    if name == "claude":
        return ClaudeEngine(**kwargs)
    raise ValueError(f"Motor desconhecido: {name}")
