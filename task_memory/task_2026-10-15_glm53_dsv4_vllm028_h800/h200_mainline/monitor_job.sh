# Usage: bash monitor_job.sh <job name> <launcher pid> <launcher wrapper> <output dir> <interval s> <max hours>
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/stepmind_env.sh"
exec /usr/bin/python3 -u "$HERE/monitor_job.py" "$@"
