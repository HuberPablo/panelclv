#!/usr/bin/env bash
# Exits when the factorial is complete AND the cdnow sign test has its 10 studies,
# or when the fleet has been empty for two consecutive checks with work outstanding.
cd /home/virthian/Desktop/Thesis/panelclv
export PYTHONPATH=src
PY=/home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python
empty=0
while true; do
  fact=$(ls -d Studies/factorial__*/ 2>/dev/null | wc -l)
  sign=$(ls Studies/selection_rescore__*cdnow*/selection_rescore.csv 2>/dev/null | wc -l)
  fleet=$($PY -c "import json,subprocess;print(len(json.loads(subprocess.run(['vastai','show','instances','--raw'],capture_output=True,text=True).stdout or '[]')))" 2>/dev/null || echo "?")
  echo "$(date +%H:%M) factorial $fact/640, cdnow sign test $sign/10, fleet $fleet"
  [ "$fact" -ge 640 ] && [ "$sign" -ge 10 ] && { echo "ALL COMPLETE"; exit 0; }
  if [ "$fleet" = "0" ]; then
    empty=$((empty+1))
    [ "$empty" -ge 2 ] && { echo "FLEET EMPTY WITH WORK OUTSTANDING"; exit 1; }
  else
    empty=0
  fi
  sleep 600
done
