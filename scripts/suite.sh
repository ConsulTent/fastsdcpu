#!/bin/bash
# usage: suite.sh <pass_name> <src_dir>
R=$RUN_DIR; P=$1; SRC=$2
mkdir -p $R/out_$P $R/logs_$P; echo "{}" > $R/settings_$P.yaml
echo -e "pass\tcase\tstatus\twall_s\tlatency_s\tloader_or_error" > $R/timings_$P.tsv
run() { name=$1; img=$2; shift 2
  t0=$(date +%s.%N)
  docker run --rm -e DEVICE=cpu -e CASE=$name -v $SRC:/app/src:ro -v $R/driver.py:/driver.py:ro -v fastsdcpu_hf:/root/.cache/huggingface -v $R/out_$P:/out -v $R/settings_$P.yaml:/app/configs/settings.yaml fastsdcpu-test python /driver.py --prompt "a cat" "$@" > $R/logs_$P/$name.log 2>&1
  t1=$(date +%s.%N); L=$R/logs_$P/$name.log
  lat=$(tr '\r' '\n' < $L | grep -m1 '^Latency' | sed 's/.*: //;s/ seconds//')
  ld=$(tr '\r' '\n' < $L | grep -m1 -E '^Loading (OpenVINO|Iris)|^\*\*\*\*\* Init (LCM|Text)' | cut -c1-90)
  err=$(tr '\r' '\n' < $L | grep -m1 '^Error in generating' | cut -c1-110)
  if [ "$img" = 1 ]; then [ -n "$lat" ] && [ -z "$err" ] && st=PASS || st=FAIL; else [ -z "$lat" ] && st=PASS || st=FAIL; fi
  echo -e "$P\t$name\t$st\t$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")\t${lat:--}\t${ld:-$err}" >> $R/timings_$P.tsv; }
OV="--use_openvino --openvino_lcm_model_id"; SZ="--image_width 512 --image_height 512"
run 1_ov_sd_turbo      1 $SZ $OV rupeshs/sd-turbo-openvino --inference_steps 1 --guidance_scale 1
run 2_ov_flux1_schnell 1 $SZ $OV OpenVINO/FLUX.1-schnell-int4-ov --inference_steps 2 --guidance_scale 0
run 3_ov_sana_sprint   1 $SZ $OV rupeshs/sana-sprint-0.6b-openvino-int4 --inference_steps 2 --guidance_scale 1
run 4_ov_flux2_klein   1 $SZ $OV rupeshs/flux2-klein-4b-int4-ov --inference_steps 4 --guidance_scale 1
run 5_lcm_stale_klein  1 $SZ --lcm_model_id stabilityai/sd-turbo --use_tiny_auto_encoder --negative_prompt blurry --inference_steps 1 --guidance_scale 1
run 6_lcmlora_stale    1 $SZ --use_lcm_lora --base_model_id Lykon/dreamshaper-8 --lcm_lora_id latent-consistency/lcm-lora-sdv1-5 --inference_steps 4 --guidance_scale 1
run 7_ov_klein_taesd   0 $SZ $OV rupeshs/flux2-klein-4b-int4-ov --use_tiny_auto_encoder --inference_steps 4
run 8_ov_flux3_klein   1 $SZ $OV rupeshs/flux3-image-klein-int4-ov --inference_steps 4 --guidance_scale 1
run 9_ov_iris_1024     1 --image_width 1024 --image_height 1024 $OV rupeshs/iris-3b-int8-ov --inference_steps 4 --guidance_scale 1
echo DONE >> $R/timings_$P.tsv
