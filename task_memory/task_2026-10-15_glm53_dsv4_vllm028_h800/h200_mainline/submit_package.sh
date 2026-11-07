# Usage: bash submit_package.sh <package dir> <launch wrapper> <scratch dir>
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/stepmind_env.sh"
exec /usr/bin/python3 -u "$HERE/submit_package.py" "$@"
