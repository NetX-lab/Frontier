import numpy as np

from frontier.profiling.utils import ProfileMethod, normalize_profile_method
from frontier.profiling.utils.singleton import Singleton


class TimerStatsStore(metaclass=Singleton):
    def __init__(self, profile_method: str, disabled: bool = False):
        self.disabled = disabled
        self.profile_method = ProfileMethod[normalize_profile_method(profile_method).upper()]
        self.TIMING_STATS = {}

    def record_time(self, name: str, time):
        name = name.replace("vidur_", "")
        if name not in self.TIMING_STATS:
            self.TIMING_STATS[name] = []

        self.TIMING_STATS[name].append(time)

    def clear_stats(self):
        self.TIMING_STATS = {}

    def get_times(self):
        """Materialize recorded samples as plain milliseconds."""

        return {
            name: [
                float(
                    time
                    if isinstance(time, float)
                    else time[0].elapsed_time(time[1])
                )
                for time in times
            ]
            for name, times in self.TIMING_STATS.items()
        }

    @staticmethod
    def get_stats_from_times(times_by_name):
        """Summarize already-materialized timing samples."""

        stats = {}
        for name, times in times_by_name.items():
            stats[name] = {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
                "count": len(times),
            }

        return stats

    def get_stats(self):
        return self.get_stats_from_times(self.get_times())
