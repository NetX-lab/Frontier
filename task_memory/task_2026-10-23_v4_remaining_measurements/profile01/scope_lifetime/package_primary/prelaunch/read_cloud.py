from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd

assert pwd.getpwuid(os.getuid()).pw_name == "i-fengyicheng"
root = Path("/mnt/ws-jfs/ycfeng")
destination = root / "frontier/glm53_dsv4_vllm028/t18/dsv4_h800_scope_lifetime_20261024_01"
assert destination.is_relative_to(root)
parent = destination.parent
assert parent.is_dir()
stat = os.statvfs(parent)
receipt = {"observed_utc": datetime.now(timezone.utc).isoformat(), "user": "i-fengyicheng", "worker_path": "/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_scope_lifetime_20261024_01", "readonly_alias": str(destination), "destination_exists": destination.exists(), "destination_lexists": os.path.lexists(destination), "parent_exists": parent.is_dir(), "parent_writable": os.access(parent, os.W_OK), "filesystem_available_bytes": stat.f_bavail * stat.f_frsize, "filesystem_free_inodes": stat.f_favail, "writes": 0, "limits": "One cloud filesystem snapshot, not reserved bytes. All cloud paths remain under the personal ycfeng directory."}
assert not receipt["destination_lexists"] and receipt["parent_writable"]
print(json.dumps(receipt, indent=2, sort_keys=True))
