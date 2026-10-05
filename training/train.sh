#!/usr/bin/env bash
# Ajuste fino do Tesseract (modelo spa_old) com as suas páginas corrigidas.
#
#   python -m guarani_ocr.cli export-gt      # gera training/ground-truth/
#   bash training/train.sh [ITERAÇÕES]       # padrão: 6000
#
# Resultado: training/output/grn_old.traineddata, já copiado para o tessdata.
# Depois use:  OCR_TESS_LANG=grn_old streamlit run app.py
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL_NAME:-grn_old}
ITER=${1:-6000}
TESSDATA=${TESSDATA_PREFIX:-$(dirname "$(find /usr/share /usr/local/share /opt/homebrew/share -name spa.traineddata 2>/dev/null | head -1)")}

[ -d tesstrain ] || git clone --depth 1 https://github.com/tesseract-ocr/tesstrain
if [ ! -f "$TESSDATA/spa_old.traineddata" ] || [ "$(stat -c%s "$TESSDATA/spa_old.traineddata" 2>/dev/null || stat -f%z "$TESSDATA/spa_old.traineddata")" -lt 1000000 ]; then
  echo "Baixando spa_old (tessdata_best) para $TESSDATA…"
  curl -fL -o "$TESSDATA/spa_old.traineddata" \
    https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/spa_old.traineddata
fi

# unicharsets de referência (o Makefile usa wget em github.com; aqui via raw)
mkdir -p tesstrain/data/langdata
SCRIPTS=$(make -s -C tesstrain --eval='print-scripts: ; @echo $(TESSERACT_SCRIPTS)' print-scripts | tail -1)
for f in radical-stroke.txt $(for s in $SCRIPTS; do echo "$s.unicharset"; done); do
  [ -s "tesstrain/data/langdata/$f" ] || curl -fsSL -o "tesstrain/data/langdata/$f" \
    "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/$f"
done

rm -rf tesstrain/data/"$MODEL"-ground-truth
mkdir -p tesstrain/data
cp -r ground-truth tesstrain/data/"$MODEL"-ground-truth

make -C tesstrain training \
  MODEL_NAME="$MODEL" \
  START_MODEL=spa_old \
  TESSDATA="$TESSDATA" \
  PSM=7 \
  RATIO_TRAIN=0.9 \
  MAX_ITERATIONS="$ITER" \
  LEARNING_RATE=0.0001

mkdir -p output
cp tesstrain/data/"$MODEL".traineddata output/
cp output/"$MODEL".traineddata "$TESSDATA"/ 2>/dev/null \
  || echo "Copie output/$MODEL.traineddata para $TESSDATA (precisa de permissão)."
echo "Pronto: modelo $MODEL instalado."
