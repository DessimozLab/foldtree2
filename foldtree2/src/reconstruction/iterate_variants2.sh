#!/usr/bin/env bash
# Second iteration batch: persistent geometry injection, and a no-geometry
# control (withhold ratio 1.0 hides structure from every known row, so the
# encoder sees no geometry at all — any gap vs the base run is the geometry
# contribution).
set -u
PY=/home/dmoi/miniforge3/envs/foldtree2/bin/python
FAMS="1003993 1003995 1007909 1013205 1016099 1026744 1027516 1029405 1032056 1032103 1033118 1033570 1041721 1044643 1045141 1049256 1049443 1054529 1057967 1058114 1058810 1058812 1058813 1059708 1066375 1066376 1067221 1067222 1067223 1067224 1067226 1067240 1067279 1067286 1067288 1067291 1067292 1067294 1067296 1067297 1067298 1067299 1067300 1067301 1067302 1067305 1067306 1067308 1067309 1067310 1067311 1067313 1067314 1067315 1067317 1067318 1067319 1067322 1067324 1067325 1067328 1067331 1067332 1067333 1067335 1067337 1067338 1067340 1067341 1067342 1067343 1067352 1067353 1067361 1067364 1067366 1067369 1067375 1067377 1067378 1067380 1067384 1067388 1067389 1067390 1067393 1067396 1067400 1067401 1067403 1067408 1067415 1067424 1067427 1067430 1067440 1067442 1067443 1067452 1067467 1067468 1067469 1067473 1067477 1067478 1067482 1067483 1067484 1067485 1067505 1067508 1067522 1067525 1067528 1067531 1067539 1067548 1067568 1067574 1067590"

run() {
  name=$1; shift
  out=models/msa_reconstruction/$name
  mkdir -p "$out"
  echo "=== starting $name: $* ==="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m foldtree2.src.reconstruction.train_msa_reconstruction \
    --family-ids $FAMS --epochs 10 --batch-size 4 --mask-ratio 0.25 \
    --max-pairs 40000 --loss-weights distance=1,angle=0.1 --patience 4 \
    --embed-dim 128 --num-layers 4 --num-heads 4 --focal-gamma 2.0 \
    --outdir "$out" --accelerator cuda --devices 1 "$@" \
    > "$out/training.log" 2>&1
  echo "=== $name results ==="
  tr '\r' '\n' < "$out/training.log" | grep -oE "val/[a-z_]+=[0-9.]+" | awk 'NR%6==0' | tail -6
  tr '\r' '\n' < "$out/training.log" | tail -2 | cut -c1-160
  echo
}

run v4_persistent --geometry-withhold-ratio 0.3 --geom-persistent
run v5_no_geom    --geometry-withhold-ratio 1.0
run v6_coord      --geometry-withhold-ratio 0.3 --coord-head --loss-weights distance=1,angle=0.1,coord=1
