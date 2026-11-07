# Usage: bash read_job.sh <job name> <output json>
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/stepmind_env.sh"
exec timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python "$HERE/read_job.py" "$@"
