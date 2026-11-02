"""Read-only retained clocks; print CPU arithmetic, never a serving gate."""
from datetime import datetime
import json
from pathlib import Path

try:
    TASK = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800')
    receipt = json.loads((TASK/'h200_mainline/diagnostic/inputs/p2_64r/platform_terminal_receipt.json').read_text())
    seconds = []
    for replica in receipt['replicas']:
        for container in replica['containers']:
            record = container['state']['terminated']
            parse = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
            elapsed = (parse(record['finishedAt'])-parse(record['startedAt'])).total_seconds()
            if elapsed != container['duration_seconds']:
                raise ValueError('Retained duration differs from terminal clocks')
            seconds.append(elapsed)
    hours = sum(seconds)*8/3600
    if hours != receipt['container_gpu_hours']:
        raise ValueError('GPU-hour arithmetic differs from receipt')
    print(json.dumps(dict(status='PASS', container_durations_s=seconds, aggregate_seconds=sum(seconds), container_gpu_hours=hours)))
except Exception as error:
    print(f"UNVERIFIED {type(error).__name__}: {error}")
