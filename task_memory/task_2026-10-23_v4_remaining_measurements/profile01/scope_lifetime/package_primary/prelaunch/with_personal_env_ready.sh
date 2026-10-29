#!/usr/bin/env bash
set -euo pipefail
set +x
export TMPDIR=/data/ycfeng/tmp
scope_proxy_script="$(curl --noproxy '*' --fail --silent --show-error --max-time 15 http://deploy.i.shaipower.com/httpproxy)"
eval "$scope_proxy_script"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
export AUTHORIZATION=D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY
export BRAINPP_ACCESS_KEY="$(cat /data/ycfeng/tmp/brainctl-auth-i-fengyicheng/accesskey_id)"
export BRAINPP_SECRET_KEY="$(cat /data/ycfeng/tmp/brainctl-auth-i-fengyicheng/accesskey_secret)"
export STEPMIND_BACKEND=rjob
export PYTHONPATH=/data/ycfeng/steptron
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export STEPMIND_PYTHON=/data/ycfeng/tmp/stepmind-env/bin/python
: "${KUBEBRAIN_WORKSPACE_NAME:?Local workspace is required}"
: "${AUTHORIZATION:?The recorded task authorization is required}"
exec "$@"
