#!/bin/bash
# Keep a training stream on a GPU box fed from a store on another machine,
# through this one (the two can't reach each other directly).
#
#   relay_shards.sh SRC_HOST:SRC_STORE DST_HOST:DST_STREAM [AHEAD]
#
# SRC_STORE holds converted shards and order-master.txt, the one order every
# run reads them in (so different net sizes see the same data). Shards that
# appear in the store and are not yet in the master order are appended to it.
#
# DST_STREAM is a train_supervised --stream directory. The relay keeps AHEAD
# (default 16) unconsumed shards present there, appends each to order.txt
# only after it has fully arrived, and deletes the ones the trainer has listed
# in consumed-*.txt. Stop it by PID; it is safe to restart.
set -u
SRC=$1 DST=$2 AHEAD=${3:-16}
SH=${SRC%%:*} SD=${SRC#*:} DH=${DST%%:*} DD=${DST#*:}
log() { echo "$(date '+%F %T') $*"; }

while :; do
  # 1. Extend the master order with newly converted shards, oldest first.
  # A shard is complete once all 8 arrays are there and no temp files are
  # (rsync and the converter both write under temporary names, then rename).
  ssh "$SH" "cd '$SD' && touch order-master.txt && for d in \$(ls -tr | grep -v -E '\.tmp$|\.txt$' \
    | grep -vxF -f order-master.txt); do [ \$(ls \$d/*.npy 2>/dev/null | wc -l) = 8 ] \
    && [ -z \"\$(ls -A \$d | grep '^\.')\" ] && echo \$d; done >> order-master.txt; cat order-master.txt" \
    > /tmp/relay-master.$$ \
    || { log "source unreachable"; sleep 60; continue; }

  # 2. What the stream already has, and what it has finished with.
  ssh "$DH" "mkdir -p '$DD' && cd '$DD' && touch order.txt && cat order.txt && echo '--' \
    && { cat consumed-*.txt 2>/dev/null; true; }" > /tmp/relay-dst.$$ || { log "dest unreachable"; sleep 60; continue; }
  listed=$(sed '/^--$/,$d' /tmp/relay-dst.$$)
  consumed=$(sed '1,/^--$/d' /tmp/relay-dst.$$)

  # 3. Delete consumed shards.
  if [ -n "$consumed" ]; then
    ssh "$DH" "cd '$DD' && for s in $(echo $consumed); do [ -d \"\$s\" ] && rm -rf -- \"\$s\" && echo deleted \$s; done" \
      | while read -r l; do log "$l"; done
  fi

  # 4. Top up: listed but unconsumed shards should number AHEAD.
  live=$(comm -23 <(echo "$listed" | grep . | sort) <(echo "$consumed" | grep . | sort) | wc -l)
  want=$(( AHEAD - live ))
  added=0
  if [ "$want" -gt 0 ]; then
    for s in $(grep -vxF -f <(echo "$listed" | grep . || echo __none__) /tmp/relay-master.$$ | head -n "$want"); do
      if ssh "$SH" "cd '$SD' && tar cf - '$s'" | ssh "$DH" "cd '$DD' && rm -rf '$s.part' && mkdir '$s.part' \
           && tar xf - -C '$s.part' && mv '$s.part/$s' '$s' && rmdir '$s.part' && echo '$s' >> order.txt"; then
        log "copied $s"; added=$((added + 1))
      else
        log "COPY FAILED $s"; break
      fi
    done
  fi
  [ "$added" -gt 0 ] || sleep 60
done
