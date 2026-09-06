#!/bin/bash
# Заливка кода проекта на под. Данные не трогаем — они живут только на поде.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/home/jovyan/shared-volume/sakha-spell
ssh -o BatchMode=yes -i ~/.ssh/id_michil hgx-2223 "mkdir -p $REMOTE/{scripts,sakhaspell,data,runs,docs}"
tar -cz -C "$HERE" scripts sakhaspell docs 2>/dev/null \
  | ssh -o BatchMode=yes -i ~/.ssh/id_michil hgx-2223 "tar -xz -C $REMOTE"
echo "залито в $REMOTE"
