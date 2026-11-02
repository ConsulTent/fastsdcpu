#!/bin/bash
R=$RUN_DIR
stub(){ t0=$(date +%s.%N); docker run --rm -i -v $2:/app/src:ro -e PYTHONPATH=/app/src -w /app fastsdcpu-test python - < $R/check_lcm_regressions.py 2>&1 | grep -E "^(PASS|FAIL)" > $R/stub_$1.txt; echo "exit ${PIPESTATUS[0]}  wall $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N)-$t0}")s" >> $R/stub_$1.txt; }
stub before $R/src_base
stub after $FIXED_SRC
$R/suite.sh before $R/src_base
$R/suite.sh after $FIXED_SRC
echo ALLDONE > $R/both.done
