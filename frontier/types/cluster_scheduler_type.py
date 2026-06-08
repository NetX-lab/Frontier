from frontier.types.base_int_enum import BaseIntEnum


class ClusterSchedulerType(BaseIntEnum):
    ROUND_ROBIN = 0
    RANDOM = 1
    LOR = 2
    STICKY_ROUND_ROBIN = 3
    STICKY_LOR = 4
