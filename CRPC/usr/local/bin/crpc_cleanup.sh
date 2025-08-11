#!/usr/bin/env bash
set -euo pipefail

# ---- Limiti directory (come prima) ----
TILES_MAX_MB=100        # /tmp/tiles + /tmp/tiles_done
RUNS_MAX_MB=500        # /tmp/yolo_runs
LOGS_MAX_MB=200         # /tmp/crpc_logs

# ---- Limiti file singoli (nuovo) ----
# Se un file supera LOGFILE_MAX_MB o SWEEP_MAX_MB, viene TRONCATO mantenendo solo gli ultimi KEEP_TAIL_MB
LOGFILE_MAX_MB=30      # per *.jsonl e *.log
SWEEP_MAX_MB=10         # per sweep_*.txt
KEEP_TAIL_MB=5         # porzione di coda da conservare

# ---- Path interessati ----
DIR_TILES="/tmp/tiles"
DIR_TILES_DONE="/tmp/tiles_done"
DIR_RUNS="/tmp/yolo_runs"
DIR_LOGS="/tmp/crpc_logs"
DIR_SWEEP="/tmp/hackrf_sweeps_text"

# ---- Funzioni ----
trim_dir_to_mb() {
  local dir="$1" max_mb="$2"
  [[ -d "$dir" ]] || return 0
  local use_kb
  use_kb=$(du -sk "$dir" | awk '{print $1}')
  local use_mb=$(( use_kb / 1024 ))
  if (( use_mb <= max_mb )); then return 0; fi
  local to_free_kb=$(( (use_mb - max_mb) * 1024 ))
  # rimuovi i file più vecchi finché scendiamo sotto il limite
  find "$dir" -type f -printf '%T@ %p\n' | sort -n | \
    awk -v tfk="$to_free_kb" '
      BEGIN{sum=0}
      {
        cmd = "du -k \""$2"\" | awk \x27{print $1}\x27"
        cmd | getline k; close(cmd)
        files[NR]=$2; sizes[NR]=k; sum+=k
        if (sum>=tfk) {
          for(i=1;i<=NR;i++) print files[i]
          exit
        }
      }' | xargs -r rm -f
}

bytes_of() { stat -c '%s' "$1" 2>/dev/null || echo 0; }

truncate_keep_tail_mb() {
  local file="$1" keep_mb="$2"
  [[ -f "$file" ]] || return 0
  # sicurezza: solo sotto /tmp
  [[ "$file" == /tmp/* ]] || return 0
  local keep_bytes=$(( keep_mb * 1024 * 1024 ))
  local size; size=$(bytes_of "$file")
  (( size > keep_bytes )) || return 0

  local tmpf
  tmpf="$(mktemp "${file}.XXXX")"
  # conserva la coda (ultimi keep_bytes)
  tail -c "$keep_bytes" "$file" > "$tmpf" || true
  # sostituzione atomica preservando permessi/ownership quanto possibile
  # (mv su stesso fs mantiene owner; i permessi verranno quelli di tmpf)
  # quindi copiamo i permessi originali
  local mode owner group
  mode=$(stat -c '%a' "$file"); owner=$(stat -c '%u' "$file"); group=$(stat -c '%g' "$file")
  mv -f "$tmpf" "$file"
  chmod "$mode" "$file" || true
  chown "$owner":"$group" "$file" || true
  echo "[cleanup] Troncato $file da $((size/1024/1024))MB a ultimi ${keep_mb}MB"
}

truncate_if_over_mb() {
  local file="$1" max_mb="$2" keep_mb="$3"
  [[ -f "$file" ]] || return 0
  local size_b; size_b=$(bytes_of "$file")
  local size_mb=$(( size_b / 1024 / 1024 ))
  if (( size_mb > max_mb )); then
    truncate_keep_tail_mb "$file" "$keep_mb"
  fi
}

# ---- 1) Trim directory (come prima) ----
trim_dir_to_mb "$DIR_TILES"       "$TILES_MAX_MB"
trim_dir_to_mb "$DIR_TILES_DONE"  "$TILES_MAX_MB"
trim_dir_to_mb "$DIR_RUNS"        "$RUNS_MAX_MB"
trim_dir_to_mb "$DIR_LOGS"        "$LOGS_MAX_MB"

# ---- 2) Troncatura file “crescenti” ----
# Sweep text
if [[ -d "$DIR_SWEEP" ]]; then
  find "$DIR_SWEEP" -maxdepth 1 -type f -name 'sweep_*.txt' -print0 | while IFS= read -r -d '' f; do
    truncate_if_over_mb "$f" "$SWEEP_MAX_MB" "$KEEP_TAIL_MB"
  done
fi

# Log jsonl & log testo
if [[ -d "$DIR_LOGS" ]]; then
  find "$DIR_LOGS" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.log' \) -print0 | while IFS= read -r -d '' f; do
    truncate_if_over_mb "$f" "$LOGFILE_MAX_MB" "$KEEP_TAIL_MB"
  done
fi

exit 0
