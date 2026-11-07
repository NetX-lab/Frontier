# Sourced by submit_package.sh, read_job.sh and monitor_job.sh: personal StepMind query environment.
set +x
export TMPDIR=/data/ycfeng/tmp
proxy_script=$(curl --noproxy '*' --fail --silent --show-error --max-time 15 http://deploy.i.shaipower.com/httpproxy)
eval "$proxy_script"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy"
export STEPMIND_BACKEND=rjob PYTHONPATH=/data/ycfeng/steptron PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export KUBEBRAIN_WORKSPACE_NAME=kun-workspace-vgen2 KUBEBRAIN_CLUSTER_ENTRY=https://platform.shaipower.com KUBEBRAIN_NAMESPACE=shai-core
