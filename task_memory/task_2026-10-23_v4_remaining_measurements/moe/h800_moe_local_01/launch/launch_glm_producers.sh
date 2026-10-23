#!/usr/bin/env bash
# Submit one GLM-5.3 producer job config and block until the job ends.
# Usage: launch_glm_producers.sh <runs/<revision>/<producer>.json>
# Keep this process alive for the whole job (run it under nohup); its exit
# handler can stop the worker.
set -euo pipefail
set +x
config=$(realpath "$1")
launch_dir=$(dirname "$(realpath "$0")")
export TMPDIR=/data/ycfeng/tmp
proxy_script=$(curl --noproxy '*' --fail --silent --show-error --max-time 15 http://deploy.i.shaipower.com/httpproxy)
eval "$proxy_script"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
export STEPMIND_BACKEND=rjob PYTHONPATH=/data/ycfeng/steptron PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
credentials=/data/ycfeng/tmp/brainctl-auth-i-fengyicheng
BRAINPP_ACCESS_KEY="$(cat "$credentials/accesskey_id")"
BRAINPP_SECRET_KEY="$(cat "$credentials/accesskey_secret")"
export BRAINPP_ACCESS_KEY BRAINPP_SECRET_KEY
exec /data/ycfeng/tmp/stepmind-env/bin/python -u "$launch_dir/glm_producer_jobs.py" submit --config "$config"
