#!/usr/bin/env bash
set -euo pipefail
set +x
export TMPDIR=/data/ycfeng/tmp
D29_PREPARATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! D29_PROXY_RECIPE=$(curl --noproxy '*' --fail --silent --show-error --max-time 15 http://deploy.i.shaipower.com/httpproxy); then
    echo "StepMind proxy environment could not be read" >&2
    exit 1
fi
eval "$D29_PROXY_RECIPE"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
D29_CREDENTIAL_DIR=/data/ycfeng/tmp/brainctl-auth-i-fengyicheng
test -s "$D29_CREDENTIAL_DIR/accesskey_id"
test -s "$D29_CREDENTIAL_DIR/accesskey_secret"
export BRAINPP_ACCESS_KEY="$(cat "$D29_CREDENTIAL_DIR/accesskey_id")"
export BRAINPP_SECRET_KEY="$(cat "$D29_CREDENTIAL_DIR/accesskey_secret")"
export STEPMIND_BACKEND=rjob
export PYTHONPATH=/data/ycfeng/steptron
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
: "${KUBEBRAIN_WORKSPACE_NAME:?Run in the authorized local workspace}"
exec /data/ycfeng/tmp/stepmind-env/bin/python -B -u "$D29_PREPARATION_DIR/launch_worker.py" \
    --plan "$D29_PREPARATION_DIR/worker_plan.json" --submit
