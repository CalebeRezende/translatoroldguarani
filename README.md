# OCR de Guarani Antigo

Extrai texto de impressos jesuíticos em guarani antigo (séc. XVII–XVIII; por
exemplo *Ara poru aguĳey haba*, de José Insaurralde) e salva num banco de dados,
com opção de sincronizar com o Google Drive. Você corrige as transcrições, e
essas correções servem para **treinar** um modelo de OCR próprio, que melhora a
cada lote de páginas.

```
imagem ──► divide página dupla ─► endireita ─► binariza ─► OCR ─► banco SQLite ─► Google Drive
                                                            │                       (pasta + Planilha)
             você corrige na interface ◄────────────────────┘
                     │
                     └─► export-gt ─► train.sh ─► modelo grn_old (Tesseract treinado)
```

## Dois motores de OCR

| Motor | Custo | Qualidade | Quando usar |
|---|---|---|---|
| **Tesseract** (`spa_old` → depois `grn_old`) | grátis, local | boa no espanhol; erra diacríticos do guarani até ser treinado | volume grande; depois de treinado |
| **Claude** (visão) | pago por página ([preços](https://www.anthropic.com/pricing)) | lê bem ſ, ỹ, g̃ e itálico sem treino | gerar transcrições iniciais boas, que viram dados de treino |

Fluxo sugerido: transcreva as primeiras 30–50 páginas com Claude (ou Tesseract),
corrija na interface, treine o Tesseract e passe a usar o `grn_old` no restante.

## Instalação

```bash
# 1. Tesseract (Ubuntu/Debian; no macOS: brew install tesseract)
sudo apt install tesseract-ocr tesseract-ocr-spa

# 2. Modelo de espanhol antigo (com ſ), ponto de partida do treino
sudo curl -L -o /usr/share/tesseract-ocr/5/tessdata/spa_old.traineddata \
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/spa_old.traineddata

# 3. Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Modelo já treinado para guarani (ver "Resultados" abaixo):

```bash
sudo cp models/grn_old.traineddata /usr/share/tesseract-ocr/5/tessdata/
export OCR_TESS_LANG=grn_old
```

Opcional — Claude: `export ANTHROPIC_API_KEY=...` (o motor aparece na interface).

## Uso

### Interface web (anexar imagem → texto)

```bash
streamlit run app.py
```

1. Aba **Nova imagem**: anexe uma ou várias imagens e clique em *Extrair texto*.
   Digitalizações de página dupla são divididas automaticamente.
2. Aba **Páginas salvas**: imagem ao lado do texto; corrija e clique em
   *Salvar correção*. **Mantenha uma linha de texto por linha impressa** — é
   isso que permite usar a página para treino.

### Linha de comando (lotes)

```bash
python -m guarani_ocr.cli ocr pasta_com_imagens/ --book "Ara poru" [--engine claude] [--drive]
python -m guarani_ocr.cli list
python -m guarani_ocr.cli correct 12 pagina12_corrigida.txt
python -m guarani_ocr.cli export-txt textos/
```

### Convenções de transcrição

- Diplomática: mantenha ſ, &, acentos graves (à, è, vì), abreviaturas, § .
- Guarani — use sempre o mesmo caractere para a mesma marca impressa
  (o modelo aprende exatamente o que você digitar):
  - marca nasal (o "chapéu" sobre a vogal) → circunflexo: `â ê î ô û ŷ`
    (ex.: `Tûpâ`, `mîrî`, `haguâ`);
  - marca da vogal gutural (a "meia-lua") → breve: `ĭ`, `y̆` (y + U+0306)
    (ex.: `ĭpĭ`, `mbohapĭ`, `aguĭyey`, `tĭbey̆`);
  - `ç`, `ñ` e acentos agudo/grave como impressos.
- Itálico entre `_sublinhados_`.
- Ilegível: `[ilegível]`; leitura duvidosa: `palavra[?]` (essas linhas ficam fora do treino).

O banco guarda também uma **versão modernizada** (ſ→s, hifenização de fim de
linha juntada, parágrafos corridos), útil para leitura e tradução.

## OCR do Google (Cloud Vision)

Terceiro motor, para comparar. Não aprende o guarani (não é treinável), mas é
útil como rascunho e como referência.

1. No Google Cloud, com a **Cloud Vision API** ativada: *Criar credenciais →
   Chave de API*. Em "Restringir chave", limite à Cloud Vision API.
2. A Vision API exige **faturamento ativado** no projeto (as primeiras 1.000
   páginas/mês são gratuitas; depois é cobrado por página).
3. ```bash
   export GOOGLE_VISION_API_KEY=...       # não coloque a chave no código nem no git
   python -m guarani_ocr.cli eval --engine google --only prologo_p1 indice_c_p1
   ```
   Isso mede o Google nas mesmas páginas de teste da tabela de resultados.
   Na interface (`streamlit run app.py`) o motor "google" aparece sozinho.

## Salvar no Google Drive

Cada página vai para uma pasta do Drive (PNG + .txt) e vira uma linha numa
**Planilha Google**, que funciona como banco de dados consultável.

1. No [Google Cloud Console](https://console.cloud.google.com/): crie um projeto,
   ative **Google Drive API** e **Google Sheets API**.
2. *APIs e serviços → Credenciais → Criar credenciais → ID do cliente OAuth →
   App para computador*. Baixe o JSON como `credentials.json` na raiz do projeto.
3. Crie uma pasta no Drive e uma Planilha Google vazia; copie os ids das URLs:
   `drive.google.com/drive/folders/<DRIVE_FOLDER_ID>` e
   `docs.google.com/spreadsheets/d/<SHEET_ID>/edit`.
4. ```bash
   export DRIVE_FOLDER_ID=...  SHEET_ID=...
   ```
   Na primeira vez o navegador abre para você autorizar; o token fica em
   `data/google_token.json`.

O banco local (`data/ocr.db`) continua sendo a fonte principal; o Drive é uma
cópia sincronizada a cada OCR e a cada correção.

## Treinar o modelo

Não existe modelo de OCR pronto para guarani (o Tesseract não tem `grn`), então
o modelo `grn_old` é treinado a partir do espanhol antigo com as suas correções.

```bash
python -m guarani_ocr.cli ocr samples/                  # OCR das páginas de exemplo
python -m guarani_ocr.cli import-gt samples/            # carrega as transcrições corrigidas (*_pN.gt.txt)
python -m guarani_ocr.cli export-gt --exclude prologo_p1 indice_c_p1   # separa 2 páginas para teste
bash training/train.sh 6000                             # ajuste fino spa_old → grn_old (~10 min)
python -m guarani_ocr.cli eval --lang spa_old --only prologo_p1 indice_c_p1
python -m guarani_ocr.cli eval --lang grn_old --only prologo_p1 indice_c_p1
export OCR_TESS_LANG=grn_old                            # passa a usar o modelo treinado
```

`--exclude`/`--only` garantem que a avaliação use páginas que o modelo nunca viu.

### Resultados até agora

Treino com 8 páginas corrigidas (6 em guarani + 2 em espanhol, 164 linhas),
teste em 2 páginas em guarani fora do treino (CER = % de caracteres errados):

| Modelo | prologo_p1 (texto corrido) | indice_c_p1 (índice) | média |
|---|---|---|---|
| `spa_old` (sem treino) | 17,6% | 12,5% | 15,3% |
| `Latin+spa_old` (sem treino) | 14,9% | 10,2% | 12,7% |
| `grn_old` (treinado, 6000 iterações) | 12,5% | 4,9% | 9,0% |

O que ainda erra: diacríticos combinados (`ângà`, `ĭù`), o `y̆` (raro nos
dados) e linhas com tinta borrada. Isso melhora com mais páginas corrigidas.

Dicas:
- Meta: ~30–50 páginas corrigidas; retreine a cada lote novo.
- Retreine sempre do `spa_old` com todo o ground truth acumulado.
- Mantenha 2–3 páginas fixas fora do treino para comparar os modelos.

## Estrutura

```
app.py                     interface web (Streamlit)
guarani_ocr/preprocess.py  divisão de página dupla, endireitamento, binarização
guarani_ocr/engines.py     Tesseract e Claude
guarani_ocr/postprocess.py limpeza e versão modernizada
guarani_ocr/storage.py     SQLite + Google Drive/Sheets
guarani_ocr/pipeline.py    fluxo imagem → banco
guarani_ocr/training.py    exportação de ground truth por linha
guarani_ocr/evaluate.py    CER
guarani_ocr/cli.py         linha de comando
training/train.sh          ajuste fino com tesstrain
models/grn_old.traineddata modelo treinado com as páginas de samples/
samples/                   páginas de exemplo + transcrições corrigidas (*.gt.txt)
```
