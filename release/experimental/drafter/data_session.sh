#!/usr/bin/env bash
# Long data session on capture kit v9: on-policy generation with decode-time capture + broad prefill capture.
set -u
cd /home/emi/code/ds41
python3 - <<'PY'
import time,os,json,sys
st=json.load(open('.state/public/current.json')); run=json.load(open(st['deployment']))
ready=f"{run['nodes'][0]['runs']}/{run['run_id']}/pair/health-ready.json"; t0=time.time()
while not os.path.exists(ready):
    if time.time()-t0>1500: sys.exit(1)
    time.sleep(10)
open('/home/emi/code/ds41/artifacts/drafter-data/session-run-id','w').write(run['run_id'])
PY
R=$(cat artifacts/drafter-data/session-run-id)
CAP=artifacts/ds41-portable-runs-v1/$R/node0/cache/drafter-capture
python3 -B release/experimental/drafter/generate.py --prompts artifacts/drafter-data/prompts-v1.jsonl --output artifacts/drafter-data/gen/gen-v2.jsonl --base http://127.0.0.1:8889 --concurrency 4 --capture-dir $CAP --data artifacts/drafter-data/capture > artifacts/drafter-data/gen/gen-v2.log 2>&1 &
python3 -B release/experimental/drafter/capture_client.py --records artifacts/drafter-data/gen/broad-v1.jsonl --capture-dir $CAP --data artifacts/drafter-data/capture --concurrency 2 > artifacts/drafter-data/gen/broad-capture.log 2>&1 &
wait
