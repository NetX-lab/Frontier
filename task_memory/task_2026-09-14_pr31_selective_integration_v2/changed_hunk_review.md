## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Initialized every production hunk against pinned main before remediation. |

# Changed-hunk review

Baseline: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`; initial candidate: `7b59d1fd7ad41f1301625f0b7c54578b5762f486`. New local hunks will be appended per package. Status is review coverage, not test execution.

| ID | File | Hunk | Disposition | Caller / rationale / evidence |
| --- | --- | --- | --- | --- |
| H001 | `frontier/attention/__init__.py` | `@@ -5,0 +6,2 @@ from frontier.attention.families import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H002 | `frontier/attention/__init__.py` | `@@ -12,0 +15 @@ from frontier.attention.model_binding import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H003 | `frontier/attention/__init__.py` | `@@ -13,0 +17 @@ from frontier.attention.model_binding import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H004 | `frontier/attention/__init__.py` | `@@ -14,0 +19 @@ from frontier.attention.model_binding import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H005 | `frontier/attention/__init__.py` | `@@ -35,0 +41,3 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H006 | `frontier/attention/__init__.py` | `@@ -37,0 +46 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H007 | `frontier/attention/__init__.py` | `@@ -42,0 +52 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H008 | `frontier/attention/families.py` | `@@ -20,0 +21 @@ _ALL_PHASES = (AttentionPhase.PREFILL, AttentionPhase.DECODE, AttentionPhase.MIX` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H009 | `frontier/attention/families.py` | `@@ -219,0 +221,83 @@ DSA_ATTENTION_FAMILY = AttentionFamilySpec(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H010 | `frontier/attention/families.py` | `@@ -224,0 +309 @@ for _family in (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H011 | `frontier/attention/gdn/__init__.py` | `@@ -0,0 +1,40 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H012 | `frontier/attention/gdn/config.py` | `@@ -0,0 +1,324 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H013 | `frontier/attention/gdn/features.py` | `@@ -0,0 +1,189 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H014 | `frontier/attention/gdn/guards.py` | `@@ -0,0 +1,48 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H015 | `frontier/attention/gdn/memory.py` | `@@ -0,0 +1,77 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H016 | `frontier/attention/gdn/state.py` | `@@ -0,0 +1,68 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H017 | `frontier/attention/model_binding.py` | `@@ -5,0 +6,5 @@ from typing import Any` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H018 | `frontier/attention/model_binding.py` | `@@ -124,0 +130,9 @@ def bind_attention_family(config: Any) -> AttentionFamilyBinding:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H019 | `frontier/attention/model_binding.py` | `@@ -185,0 +200,90 @@ def bind_attention_family(config: Any) -> AttentionFamilyBinding:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H020 | `frontier/attention/ops.py` | `@@ -27,0 +28 @@ class AttentionMemoryLayout(Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H021 | `frontier/attention/profiling_mapping.py` | `@@ -274,0 +275,14 @@ def get_enabled_predictor_feature_columns(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H022 | `frontier/attention/profiling_mapping.py` | `@@ -317,0 +332,12 @@ def get_enabled_shared_predictor_feature_columns(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H023 | `frontier/config/config.py` | `@@ -2084,0 +2085,14 @@ class ReplicaConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H024 | `frontier/config/config.py` | `@@ -2140,0 +2155,4 @@ class BaseExecutionTimePredictorConfig(BasePolyConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H025 | `frontier/config/config.py` | `@@ -5304,0 +5323 @@ class SimulationConfig(ABC):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H026 | `frontier/config/config.py` | `@@ -5342,0 +5362,43 @@ class SimulationConfig(ABC):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H027 | `frontier/config/device_sku_config.py` | `@@ -13,0 +14 @@ class BaseDeviceSKUConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H028 | `frontier/config/device_sku_config.py` | `@@ -96,0 +98,13 @@ class RtxPro6000DeviceSKUConfig(BaseDeviceSKUConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H029 | `frontier/config/model_config.py` | `@@ -7 +7,7 @@ import os` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H030 | `frontier/config/model_config.py` | `@@ -30 +36 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H031 | `frontier/config/model_config.py` | `@@ -40,0 +47 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H032 | `frontier/config/model_config.py` | `@@ -57 +64 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H033 | `frontier/config/model_config.py` | `@@ -63,0 +71,12 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H034 | `frontier/config/model_config.py` | `@@ -117,0 +137,4 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H035 | `frontier/config/model_config.py` | `@@ -142,0 +166,20 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H036 | `frontier/config/model_config.py` | `@@ -144,2 +187,2 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H037 | `frontier/config/model_config.py` | `@@ -148 +191,8 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H038 | `frontier/config/model_config.py` | `@@ -158,0 +209 @@ class QuantizationConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H039 | `frontier/config/model_config.py` | `@@ -266,0 +318 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H040 | `frontier/config/model_config.py` | `@@ -292,0 +345,11 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H041 | `frontier/config/model_config.py` | `@@ -311,0 +375,3 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H042 | `frontier/config/model_config.py` | `@@ -318,0 +385,10 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H043 | `frontier/config/model_config.py` | `@@ -427,2 +503,63 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H044 | `frontier/config/model_config.py` | `@@ -619,0 +757,6 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H045 | `frontier/config/model_config.py` | `@@ -685,0 +829,13 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H046 | `frontier/config/model_config.py` | `@@ -708,0 +865 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H047 | `frontier/config/model_config.py` | `@@ -710,0 +868 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H048 | `frontier/config/model_config.py` | `@@ -723,0 +882,8 @@ class BaseModelConfig(BaseFixedConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H049 | `frontier/config/node_sku_config.py` | `@@ -102,0 +103,12 @@ class H20DgxNodeSKUConfig(BaseNodeSKUConfig):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H050 | `frontier/config/quantization_manager.py` | `@@ -74,0 +75,8 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H051 | `frontier/config/quantization_manager.py` | `@@ -102,0 +111 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H052 | `frontier/config/quantization_manager.py` | `@@ -168,0 +178 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H053 | `frontier/config/quantization_manager.py` | `@@ -179,0 +190,9 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H054 | `frontier/config/quantization_manager.py` | `@@ -202 +221,14 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H055 | `frontier/config/quantization_manager.py` | `@@ -211,0 +244,14 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H056 | `frontier/config/quantization_manager.py` | `@@ -216,5 +262 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H057 | `frontier/config/quantization_manager.py` | `@@ -452,0 +495,2 @@ class QuantizationManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H058 | `frontier/config/utils.py` | `@@ -108,0 +109,6 @@ def dataclass_to_dict(obj):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H059 | `frontier/entities/__init__.py` | `@@ -4,0 +5 @@ from frontier.entities.execution_time import ExecutionTime` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H060 | `frontier/entities/__init__.py` | `@@ -19,0 +21 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H061 | `frontier/entities/execution_time.py` | `@@ -2 +2 @@ from collections.abc import Mapping` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H062 | `frontier/entities/execution_time.py` | `@@ -97,0 +98,3 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H063 | `frontier/entities/execution_time.py` | `@@ -100,0 +104,26 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H064 | `frontier/entities/execution_time.py` | `@@ -101,0 +131,13 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H065 | `frontier/entities/execution_time.py` | `@@ -448,0 +491,11 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H066 | `frontier/entities/execution_time.py` | `@@ -470,0 +524 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H067 | `frontier/entities/execution_time.py` | `@@ -476,0 +531 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H068 | `frontier/entities/execution_time.py` | `@@ -478,0 +534 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H069 | `frontier/entities/execution_time.py` | `@@ -496,0 +553 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H070 | `frontier/entities/execution_time.py` | `@@ -504 +561 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H071 | `frontier/entities/execution_time.py` | `@@ -506,0 +564,6 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H072 | `frontier/entities/execution_time.py` | `@@ -584,0 +648 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H073 | `frontier/entities/execution_time.py` | `@@ -592,0 +657 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H074 | `frontier/entities/execution_time.py` | `@@ -607,0 +673 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H075 | `frontier/entities/execution_time.py` | `@@ -617,0 +684 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H076 | `frontier/entities/execution_time.py` | `@@ -627,0 +695 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H077 | `frontier/entities/execution_time.py` | `@@ -637,0 +706 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H078 | `frontier/entities/execution_time.py` | `@@ -653,0 +723 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H079 | `frontier/entities/execution_time.py` | `@@ -661,0 +732 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H080 | `frontier/entities/execution_time.py` | `@@ -674,0 +746 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H081 | `frontier/entities/execution_time.py` | `@@ -678,0 +751 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H082 | `frontier/entities/execution_time.py` | `@@ -692,0 +766 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H083 | `frontier/entities/execution_time.py` | `@@ -723,0 +798 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H084 | `frontier/entities/execution_time.py` | `@@ -737 +812 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H085 | `frontier/entities/execution_time.py` | `@@ -751 +826 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H086 | `frontier/entities/execution_time.py` | `@@ -763 +838 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H087 | `frontier/entities/execution_time.py` | `@@ -1043,2 +1118,116 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H088 | `frontier/entities/execution_time.py` | `@@ -1080 +1269 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H089 | `frontier/entities/execution_time.py` | `@@ -1117 +1306 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H090 | `frontier/entities/execution_time.py` | `@@ -1320 +1509 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H091 | `frontier/entities/execution_time.py` | `@@ -1454 +1643 @@ class ExecutionTime(BaseEntity):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H092 | `frontier/entities/stage_execution_time.py` | `@@ -0,0 +1,568 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H093 | `frontier/execution_time_predictor/attention_tp_policy.py` | `@@ -33,2 +33,2 @@ def get_attention_linear_tp_policy_ops() -> frozenset[str]:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H094 | `frontier/execution_time_predictor/gdn_predictor.py` | `@@ -0,0 +1,261 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H095 | `frontier/execution_time_predictor/measurement_input_paths.py` | `@@ -0,0 +1,116 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H096 | `frontier/execution_time_predictor/random_forrest_execution_time_predictor.py` | `@@ -88,3 +88,10 @@ class RandomForrestExecutionTimePredictor:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H097 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -38,0 +39,3 @@ from frontier.execution_time_predictor.cache_io import atomic_pickle_dump` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H098 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -420,0 +424 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H099 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -426,0 +431 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H100 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -428,0 +434 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H101 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -430,0 +437 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H102 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -434,0 +442 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H103 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -436,0 +445 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H104 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -534,0 +544,2 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H105 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -538,0 +550,22 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H106 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -566 +599 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H107 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -567,0 +601,5 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H108 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -570 +608 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H109 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -572 +610 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H110 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -576 +614 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H111 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -580 +618 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H112 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -588 +626 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H113 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -591,2 +629,2 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H114 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -598,43 +636,15 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H115 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -731 +741,3 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H116 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -732,0 +745,6 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H117 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -749 +767,9 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H118 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -833,0 +860,55 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H119 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -2122 +2203 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H120 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -2220 +2301 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H121 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -2252 +2333 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H122 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -2291,0 +2373,6 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H123 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4101,0 +4189 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H124 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4116,0 +4205 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H125 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4129,0 +4219 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H126 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4144,0 +4235 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H127 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4357 +4448 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H128 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4372,0 +4464,4 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H129 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4380,0 +4476,4 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H130 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4440 +4539 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H131 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4443,0 +4543,12 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H132 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4449,0 +4561,7 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H133 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4451,2 +4569,2 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H134 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4454,0 +4573 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H135 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4460,2 +4579,2 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H136 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4465,0 +4585 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H137 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4468 +4588 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H138 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4484 +4604 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H139 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4513,0 +4634,23 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H140 | `frontier/execution_time_predictor/shared_prediction_model_manager.py` | `@@ -4517,0 +4661,3 @@ class ExecutionTimePredictionModelManager:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H141 | `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | `@@ -4 +3,0 @@ from dataclasses import dataclass` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H142 | `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | `@@ -24 +23 @@ from frontier.config.parallel_semantics import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H143 | `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | `@@ -39 +38,5 @@ from frontier.execution_time_predictor.shared_prediction_model_manager import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H144 | `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | `@@ -506,32 +509,9 @@ class SklearnDisaggregationExecutionTimePredictor(SklearnMoEExecutionTimePredict` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H145 | `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py` | `@@ -1376,0 +1357,37 @@ class SklearnDisaggregationExecutionTimePredictor(SklearnMoEExecutionTimePredict` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H146 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -33,0 +34 @@ from frontier.attention.families import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H147 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -36 +37,4 @@ from frontier.attention.families import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H148 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -72,0 +77,3 @@ from frontier.execution_time_predictor.shared_prediction_model_manager import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H149 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -133 +140 @@ from frontier.execution_time_predictor.profiling_metadata import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H150 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -370 +377 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H151 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -425,0 +433,6 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H152 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -540,0 +554 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H153 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -542,0 +557 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H154 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -559,0 +575,3 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H155 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -562,5 +580,4 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H156 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -571,2 +588,6 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H157 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -578,0 +600,3 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H158 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -730,0 +755,12 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H159 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -762,0 +799 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H160 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -769,0 +807,3 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H161 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -797,0 +838,9 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H162 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -805,30 +854,15 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H163 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -839,0 +874,2 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H164 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -843,0 +880,31 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H165 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -858,0 +926 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H166 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -866 +934 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H167 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -869 +937 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H168 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -873,2 +941,2 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H169 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -878,0 +947 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H170 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -881,4 +950 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H171 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -887 +953 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H172 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -892 +958 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H173 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -914,0 +981 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H174 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -918 +985 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H175 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -936 +1003 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H176 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -943 +1010 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H177 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -952,0 +1020,6 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H178 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -1157,5 +1230,9 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H179 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -2908,0 +2986,2 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H180 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -3498 +3577 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H181 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -4009 +4088 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H182 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -4021 +4100 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H183 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -7366 +7445,24 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H184 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -7941 +8043,7 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H185 | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | `@@ -8258,57 +8366,5 @@ class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H186 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -4 +4,4 @@ import os` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H187 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -10,0 +14 @@ from frontier.attention.families import DENSE_ATTENTION_FAMILY` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H188 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -15 +19 @@ from frontier.attention.profiling_mapping import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H189 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -46,0 +51 @@ from frontier.moe_ep_workload import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H190 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -251,0 +257,10 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H191 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -690,0 +706 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H192 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -695,0 +712,7 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H193 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -797,29 +820,6 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H194 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -845,2 +845,5 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H195 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -864,0 +868,24 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H196 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -866 +893 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H197 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -913,0 +941,15 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H198 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -917,0 +960,2 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H199 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -919,0 +964,13 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H200 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -928,2 +985,2 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H201 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -936,0 +994,4 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H202 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2155,0 +2217,255 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H203 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2166,0 +2483 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H204 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2212 +2529 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H205 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2215,0 +2533 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H206 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2424,0 +2743,9 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H207 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -2477,0 +2805 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H208 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3181 +3509,2 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H209 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3216,0 +3546,36 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H210 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3243 +3608 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H211 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3249,0 +3615,11 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H212 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3316,0 +3693 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H213 | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | `@@ -3455,68 +3832,5 @@ class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H214 | `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` | `@@ -3,0 +4 @@ from frontier.attention.memory import get_attention_runtime_kv_layout` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H215 | `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` | `@@ -52,0 +54 @@ class AnalyticalKVCacheTransferPredictor(BaseKVCacheTransferPredictor):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H216 | `frontier/metrics/constants.py` | `@@ -22,0 +23,4 @@ class OperationMetrics(enum.Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H217 | `frontier/metrics/metrics_store.py` | `@@ -11 +11 @@ from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H218 | `frontier/metrics/metrics_store.py` | `@@ -15,0 +16 @@ from frontier.attention.families import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H219 | `frontier/metrics/metrics_store.py` | `@@ -565,2 +566,5 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H220 | `frontier/metrics/metrics_store.py` | `@@ -687 +691,7 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H221 | `frontier/metrics/metrics_store.py` | `@@ -695 +705,13 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H222 | `frontier/metrics/metrics_store.py` | `@@ -987,0 +1010,143 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H223 | `frontier/metrics/metrics_store.py` | `@@ -3524,0 +3690,84 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H224 | `frontier/metrics/metrics_store.py` | `@@ -3623 +3872,7 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H225 | `frontier/metrics/metrics_store.py` | `@@ -3646 +3901 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H226 | `frontier/metrics/metrics_store.py` | `@@ -3648,0 +3904,12 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H227 | `frontier/metrics/metrics_store.py` | `@@ -3859,0 +4127,50 @@ class MetricsStore:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H228 | `frontier/metrics/op_trace_utils.py` | `@@ -533,0 +534,14 @@ def compute_op_trace_meta(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H229 | `frontier/model_architectures.py` | `@@ -19 +19 @@ logger = logging.getLogger(__name__)` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H230 | `frontier/model_architectures.py` | `@@ -345 +345 @@ def _default_layer_contracts() -> tuple[LayerContractSpec, ...]:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H231 | `frontier/model_architectures.py` | `@@ -348 +348 @@ class LinearAttentionProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H232 | `frontier/model_architectures.py` | `@@ -413 +413 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H233 | `frontier/model_architectures.py` | `@@ -424,0 +425,2 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H234 | `frontier/model_architectures.py` | `@@ -442 +444 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H235 | `frontier/model_architectures.py` | `@@ -446 +448 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H236 | `frontier/model_architectures.py` | `@@ -452,0 +455,12 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H237 | `frontier/model_architectures.py` | `@@ -486,2 +500,2 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H238 | `frontier/model_architectures.py` | `@@ -507,2 +521,2 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H239 | `frontier/model_architectures.py` | `@@ -538,2 +552,2 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H240 | `frontier/model_architectures.py` | `@@ -594,0 +609,34 @@ class ModelArchitectureProfile:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H241 | `frontier/model_architectures.py` | `@@ -1007,0 +1056,62 @@ def _matches_step3_text(config: Any) -> bool:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H242 | `frontier/model_architectures.py` | `@@ -1012,0 +1123 @@ class ModelArchitectureRegistry:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H243 | `frontier/model_architectures.py` | `@@ -1037 +1148,2 @@ class ModelArchitectureRegistry:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H244 | `frontier/model_architectures.py` | `@@ -1040 +1152 @@ class ModelArchitectureRegistry:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H245 | `frontier/model_architectures.py` | `@@ -1047,0 +1160,15 @@ class ModelArchitectureRegistry:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H246 | `frontier/model_architectures.py` | `@@ -1052,0 +1180 @@ for _profile in (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H247 | `frontier/moe_ep_workload.py` | `@@ -12 +12 @@ from collections.abc import Mapping` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H248 | `frontier/moe_ep_workload.py` | `@@ -18,0 +19,2 @@ from typing import TypeAlias` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H249 | `frontier/moe_ep_workload.py` | `@@ -24,0 +27,54 @@ RoutingDetails: TypeAlias = Mapping[int, Mapping[int, Mapping[int, Real]]]` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H250 | `frontier/moe_ep_workload.py` | `@@ -90,0 +147,5 @@ class LayerEPWorkload:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H251 | `frontier/moe_ep_workload.py` | `@@ -280,0 +342,14 @@ class LayerEPWorkload:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H252 | `frontier/moe_ep_workload.py` | `@@ -291,9 +366 @@ class LayerEPWorkload:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H253 | `frontier/moe_ep_workload.py` | `@@ -781,0 +849 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H254 | `frontier/operators/binding.py` | `@@ -158,2 +158,2 @@ def _resolve_architecture_linear_tp_mode(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H255 | `frontier/operators/binding.py` | `@@ -161 +161 @@ def _resolve_architecture_linear_tp_mode(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H256 | `frontier/operators/binding.py` | `@@ -164,2 +164,2 @@ def _resolve_architecture_linear_tp_mode(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H257 | `frontier/operators/typed_contracts.py` | `@@ -135 +135 @@ def validate_typed_operator_metadata(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H258 | `frontier/operators/typed_contracts.py` | `@@ -137 +137 @@ def validate_typed_operator_metadata(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H259 | `frontier/operators/typed_contracts.py` | `@@ -193,4 +193,4 @@ def validate_typed_operator_metadata(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H260 | `frontier/operators/typed_contracts.py` | `@@ -198,5 +198,5 @@ def validate_typed_operator_metadata(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H261 | `frontier/profiling/attention/backends/__init__.py` | `@@ -10,0 +11 @@ class AttentionBackend(Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H262 | `frontier/profiling/attention/backends/__init__.py` | `@@ -52,0 +54,6 @@ def get_attention_wrapper():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H263 | `frontier/profiling/attention/backends/__init__.py` | `@@ -75,0 +83,6 @@ def __getattr__(name: str):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H264 | `frontier/profiling/attention/backends/__init__.py` | `@@ -88,0 +102 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H265 | `frontier/profiling/attention/backends/vllm_rocm_attention_wrapper.py` | `@@ -0,0 +1,329 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H266 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -10,0 +11 @@ from frontier.profiling.collectives.collectives_wrapper import CollectiveWrapper` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H267 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -14,0 +16,15 @@ logger = init_logger(__name__)` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H268 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -20 +36 @@ class BenchmarkRunner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H269 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -25,3 +41,4 @@ class BenchmarkRunner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H270 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -35,0 +53 @@ class BenchmarkRunner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H271 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -73,0 +92 @@ class BenchmarkRunner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H272 | `frontier/profiling/collectives/benchmark_runner.py` | `@@ -86 +105 @@ class BenchmarkRunner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H273 | `frontier/profiling/collectives/collectives_impl.py` | `@@ -18,0 +19 @@ class GraphedCollective:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H274 | `frontier/profiling/collectives/collectives_impl.py` | `@@ -103,0 +105,4 @@ class GraphedCollective:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H275 | `frontier/profiling/collectives/collectives_input.py` | `@@ -3,0 +4,3 @@ from random import randint` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H276 | `frontier/profiling/collectives/collectives_input.py` | `@@ -12,0 +16 @@ class CollectivesInput:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H277 | `frontier/profiling/collectives/collectives_input.py` | `@@ -17,0 +22,7 @@ class CollectivesInput:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H278 | `frontier/profiling/collectives/collectives_input.py` | `@@ -35,4 +46,4 @@ class CollectivesInput:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H279 | `frontier/profiling/collectives/collectives_wrapper.py` | `@@ -23,0 +24 @@ class CollectiveWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H280 | `frontier/profiling/collectives/collectives_wrapper.py` | `@@ -34 +35,5 @@ class CollectiveWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H281 | `frontier/profiling/collectives/collectives_wrapper.py` | `@@ -64 +69 @@ class CollectiveWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H282 | `frontier/profiling/collectives/main.py` | `@@ -0,0 +1,4 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H283 | `frontier/profiling/collectives/main.py` | `@@ -3,0 +8,2 @@ import os` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H284 | `frontier/profiling/collectives/main.py` | `@@ -6 +11,0 @@ import pandas as pd` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H285 | `frontier/profiling/collectives/main.py` | `@@ -9 +14,8 @@ from tqdm import tqdm` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H286 | `frontier/profiling/collectives/main.py` | `@@ -11 +23,2 @@ from frontier.logger import init_logger` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H287 | `frontier/profiling/collectives/main.py` | `@@ -15,0 +29,2 @@ logger = init_logger(__name__)` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H288 | `frontier/profiling/collectives/main.py` | `@@ -18 +33,12 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H289 | `frontier/profiling/collectives/main.py` | `@@ -47 +73 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H290 | `frontier/profiling/collectives/main.py` | `@@ -54 +80 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H291 | `frontier/profiling/collectives/main.py` | `@@ -59,2 +85,4 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H292 | `frontier/profiling/collectives/main.py` | `@@ -61,0 +90,6 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H293 | `frontier/profiling/collectives/main.py` | `@@ -65,3 +99,2 @@ def parse_args():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H294 | `frontier/profiling/collectives/main.py` | `@@ -69 +102,12 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H295 | `frontier/profiling/collectives/main.py` | `@@ -71,2 +114,0 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H296 | `frontier/profiling/collectives/main.py` | `@@ -74 +116,7 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H297 | `frontier/profiling/collectives/main.py` | `@@ -76,2 +123,0 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H298 | `frontier/profiling/collectives/main.py` | `@@ -78,0 +125,87 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H299 | `frontier/profiling/collectives/main.py` | `@@ -84,3 +217 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H300 | `frontier/profiling/collectives/main.py` | `@@ -92,2 +223,11 @@ def create_runner_pool():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H301 | `frontier/profiling/collectives/main.py` | `@@ -95 +234,0 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H302 | `frontier/profiling/collectives/main.py` | `@@ -97 +236,5 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H303 | `frontier/profiling/collectives/main.py` | `@@ -99 +241,0 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H304 | `frontier/profiling/collectives/main.py` | `@@ -101,6 +243,8 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H305 | `frontier/profiling/collectives/main.py` | `@@ -107,0 +252,5 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H306 | `frontier/profiling/collectives/main.py` | `@@ -109,12 +257,0 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H307 | `frontier/profiling/collectives/main.py` | `@@ -122,2 +259,2 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H308 | `frontier/profiling/collectives/main.py` | `@@ -125,2 +262,18 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H309 | `frontier/profiling/collectives/main.py` | `@@ -128,4 +281,7 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H310 | `frontier/profiling/collectives/main.py` | `@@ -133 +289,4 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H311 | `frontier/profiling/collectives/main.py` | `@@ -135,2 +294,2 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H312 | `frontier/profiling/common/accelerator.py` | `@@ -0,0 +1,232 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H313 | `frontier/profiling/common/constants.py` | `@@ -32,0 +33,4 @@ class OperationMetrics(enum.Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H314 | `frontier/profiling/common/cuda_timer.py` | `@@ -1 +1 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H315 | `frontier/profiling/common/cuda_timer.py` | `@@ -3,2 +3 @@ import time` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H316 | `frontier/profiling/common/cuda_timer.py` | `@@ -6,2 +4,0 @@ from torch.profiler import record_function` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H317 | `frontier/profiling/common/cuda_timer.py` | `@@ -8,0 +6,2 @@ from frontier.profiling.utils import ProfileMethod` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H318 | `frontier/profiling/common/cuda_timer.py` | `@@ -10,101 +9,7 @@ from frontier.profiling.utils import ProfileMethod` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H319 | `frontier/profiling/common/device_timer.py` | `@@ -0,0 +1,129 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H320 | `frontier/profiling/common/layers/layernorm.py` | `@@ -8,0 +9 @@ from frontier.profiling.common.cuda_timer import CudaTimer` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H321 | `frontier/profiling/common/layers/layernorm.py` | `@@ -9,0 +11 @@ from frontier.profiling.common.cuda_timer import CudaTimer` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H322 | `frontier/profiling/common/layers/layernorm.py` | `@@ -19,2 +21,10 @@ except ImportError:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H323 | `frontier/profiling/common/layers/layernorm.py` | `@@ -40 +50,7 @@ class RMSNorm(nn.Module):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H324 | `frontier/profiling/common/layers/layernorm.py` | `@@ -43,0 +60,6 @@ class RMSNorm(nn.Module):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H325 | `frontier/profiling/common/layers/layernorm.py` | `@@ -53,0 +76,2 @@ class RMSNorm(nn.Module):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H326 | `frontier/profiling/common/layers/layernorm.py` | `@@ -78 +102,2 @@ class GemmaRMSNorm(nn.Module):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H327 | `frontier/profiling/common/layers/rotary_embedding.py` | `@@ -25,0 +26 @@ import os` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H328 | `frontier/profiling/common/layers/rotary_embedding.py` | `@@ -31,0 +33 @@ from frontier.profiling.common.timer_stats_store import TimerStatsStore` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H329 | `frontier/profiling/common/layers/rotary_embedding.py` | `@@ -99,0 +102,50 @@ def _load_vllm_get_rope():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H330 | `frontier/profiling/common/layers/rotary_embedding.py` | `@@ -579 +631,2 @@ def get_rope(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H331 | `frontier/profiling/common/model_config.py` | `@@ -6 +6,9 @@ from typing import Any, Dict, List, Optional` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H332 | `frontier/profiling/common/model_config.py` | `@@ -53,0 +62 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H333 | `frontier/profiling/common/model_config.py` | `@@ -67,0 +77,8 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H334 | `frontier/profiling/common/model_config.py` | `@@ -123,0 +141,5 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H335 | `frontier/profiling/common/model_config.py` | `@@ -146,0 +169,19 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H336 | `frontier/profiling/common/model_config.py` | `@@ -241 +282,7 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H337 | `frontier/profiling/common/model_config.py` | `@@ -243,0 +291,53 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H338 | `frontier/profiling/common/model_config.py` | `@@ -308,0 +409 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H339 | `frontier/profiling/common/model_config.py` | `@@ -352,0 +454,3 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H340 | `frontier/profiling/common/model_config.py` | `@@ -379,0 +484,7 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H341 | `frontier/profiling/common/model_config.py` | `@@ -382,0 +494,3 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H342 | `frontier/profiling/common/model_config.py` | `@@ -556,0 +671 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H343 | `frontier/profiling/common/model_config.py` | `@@ -570,0 +686,8 @@ class ModelConfig:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H344 | `frontier/profiling/common/timer_stats_store.py` | `@@ -23,6 +23,11 @@ class TimerStatsStore(metaclass=Singleton):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H345 | `frontier/profiling/common/timer_stats_store.py` | `@@ -29,0 +35,2 @@ class TimerStatsStore(metaclass=Singleton):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H346 | `frontier/profiling/common/timer_stats_store.py` | `@@ -30,0 +38,6 @@ class TimerStatsStore(metaclass=Singleton):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H347 | `frontier/profiling/common/timer_stats_store.py` | `@@ -39 +51,0 @@ class TimerStatsStore(metaclass=Singleton):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H348 | `frontier/profiling/common/timer_stats_store.py` | `@@ -40,0 +53,3 @@ class TimerStatsStore(metaclass=Singleton):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H349 | `frontier/profiling/common/vllm_compat.py` | `@@ -0,0 +1,34 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H350 | `frontier/profiling/experimental/__init__.py` | `@@ -0,0 +1 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H351 | `frontier/profiling/experimental/sglang/__init__.py` | `@@ -0,0 +1,8 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H352 | `frontier/profiling/experimental/sglang/attention.py` | `@@ -0,0 +1,277 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H353 | `frontier/profiling/experimental/sglang/dense.py` | `@@ -0,0 +1,144 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H354 | `frontier/profiling/experimental/sglang/gdn.py` | `@@ -0,0 +1,178 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H355 | `frontier/profiling/experimental/sglang/gdn_trace.py` | `@@ -0,0 +1,318 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H356 | `frontier/profiling/experimental/sglang/graph_replay.py` | `@@ -0,0 +1,494 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H357 | `frontier/profiling/experimental/sglang/moe.py` | `@@ -0,0 +1,334 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H358 | `frontier/profiling/experimental/sglang/routed_moe_replay.py` | `@@ -0,0 +1,305 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H359 | `frontier/profiling/gdn/README.md` | `@@ -0,0 +1,22 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H360 | `frontier/profiling/gdn/__init__.py` | `@@ -0,0 +1,20 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H361 | `frontier/profiling/gdn/inputs.py` | `@@ -0,0 +1,255 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H362 | `frontier/profiling/gdn/main.py` | `@@ -0,0 +1,136 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H363 | `frontier/profiling/gdn/vllm_wrapper.py` | `@@ -0,0 +1,639 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H364 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -26 +26 @@ from frontier.model_architectures import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H365 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -66 +66,2 @@ def _uses_gemma_rms_norm(config: ModelConfig) -> bool:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H366 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -702 +703 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H367 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -705 +706 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H368 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -707 +708 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H369 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -709 +710 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H370 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -713 +714 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H371 | `frontier/profiling/linear_op/linear_op_impl.py` | `@@ -717,2 +718,2 @@ def build_linear_op_attention_module(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H372 | `frontier/profiling/linear_op/linear_op_wrapper.py` | `@@ -105 +105 @@ class LinearOpWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H373 | `frontier/profiling/linear_op/main.py` | `@@ -66,0 +67,4 @@ from frontier.profiling.common.model_config import ModelConfig` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H374 | `frontier/profiling/linear_op/main.py` | `@@ -94,50 +98,2 @@ def _get_available_gpus(num_gpus: int) -> List[int]:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H375 | `frontier/profiling/linear_op/main.py` | `@@ -790 +745,0 @@ def profile_model(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H376 | `frontier/profiling/linear_op/main.py` | `@@ -791,0 +747 @@ def profile_model(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H377 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -7 +7 @@ from typing import Dict, List, Mapping, Sequence` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H378 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -250,3 +250,3 @@ def _typed_operator_contracts(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H379 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -259,2 +259,2 @@ def _typed_operator_contracts(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H380 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -430 +430 @@ def build_profiling_plan(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H381 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -440 +440 @@ def build_profiling_plan(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H382 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -464 +464 @@ def build_profiling_plan(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H383 | `frontier/profiling/linear_op/profiling_plan.py` | `@@ -476 +476 @@ def build_profiling_plan(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H384 | `frontier/profiling/moe/main.py` | `@@ -59,0 +60,4 @@ from frontier.profiling.common.model_config import ModelConfig` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H385 | `frontier/profiling/moe/main.py` | `@@ -96 +100 @@ def _worker_init(gpu_id: int) -> None:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H386 | `frontier/profiling/moe/main.py` | `@@ -859 +863 @@ def profile_model(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H387 | `frontier/profiling/moe/main.py` | `@@ -919,53 +923,2 @@ def _get_available_gpus(num_gpus: int) -> List[int]:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H388 | `frontier/profiling/moe/moe_impl.py` | `@@ -26,0 +27 @@ from frontier.profiling.common.utils import raise_if_fp8_requested` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H389 | `frontier/profiling/moe/moe_impl.py` | `@@ -28,8 +29,30 @@ try:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H390 | `frontier/profiling/moe/moe_impl.py` | `@@ -133,6 +156,13 @@ class MoEGatingNetwork(nn.Module):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H391 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -13,2 +13,3 @@ Design rationale:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H392 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -22 +23,2 @@ import triton.language as tl` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H393 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -27,0 +30,76 @@ FP8_QUANT_AVAILABLE = False` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H394 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -33,8 +111,17 @@ try:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H395 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -42 +129 @@ try:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H396 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -337,0 +425,132 @@ def _run_fused_moe_iteration(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H397 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -398,0 +618 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H398 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -425,0 +646 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H399 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -438,0 +660,4 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H400 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -444,0 +670,5 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H401 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -473 +703 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H402 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -476,14 +706,17 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H403 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -494,0 +728,82 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H404 | `frontier/profiling/moe/moe_vllm_kernel.py` | `@@ -495,0 +811 @@ def profile_fused_moe_kernel(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H405 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -33,0 +34 @@ from frontier.profiling.moe.moe_vllm_kernel import (` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H406 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -47,0 +49,12 @@ ACTIVE_STEPS = 20` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H407 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -117 +130,8 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H408 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -119 +139,2 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H409 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -130,0 +152,2 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H410 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -591 +614,4 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H411 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -596 +622,4 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H412 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -615,0 +645 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H413 | `frontier/profiling/moe/moe_wrapper.py` | `@@ -643,0 +674 @@ class MoEWrapper:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H414 | `frontier/profiling/utils/__init__.py` | `@@ -30,0 +31 @@ class ProfileMethod(enum.Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H415 | `frontier/profiling/utils/__init__.py` | `@@ -38,0 +40 @@ EXPORTABLE_PROFILE_METHOD_CHOICES = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H416 | `frontier/profiling/utils/__init__.py` | `@@ -56,0 +59,2 @@ def normalize_profile_method(profile_method: str) -> str:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H417 | `frontier/profiling/utils/__init__.py` | `@@ -65,0 +70,2 @@ def profile_method_to_measurement_type(profile_method: str) -> MeasurementType:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H418 | `frontier/profiling/utils/__init__.py` | `@@ -67,2 +73,2 @@ def profile_method_to_measurement_type(profile_method: str) -> MeasurementType:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H419 | `frontier/profiling/utils/__init__.py` | `@@ -72,0 +79,16 @@ def profile_method_to_measurement_type(profile_method: str) -> MeasurementType:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H420 | `frontier/profiling/utils/__init__.py` | `@@ -122,0 +145,2 @@ def build_profile_method_output_path(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H421 | `frontier/profiling/utils/__init__.py` | `@@ -492,0 +517 @@ def get_collectives_inputs(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H422 | `frontier/profiling/utils/__init__.py` | `@@ -500 +525 @@ def get_collectives_inputs(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H423 | `frontier/profiling/utils/__init__.py` | `@@ -509 +534,5 @@ def get_collectives_inputs(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H424 | `frontier/profiling/utils/confirmation.py` | `@@ -176 +176 @@ def build_linear_op_config_sections(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H425 | `frontier/profiling/utils/confirmation.py` | `@@ -178 +178 @@ def build_linear_op_config_sections(` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H426 | `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | `@@ -64,0 +65,5 @@ class BaseReplicaScheduler(ABC):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H427 | `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | `@@ -387,0 +393,20 @@ class BaseReplicaScheduler(ABC):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H428 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -28,0 +29,2 @@ from frontier.config import global_vars` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H429 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -129,0 +132,18 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H430 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -1403,0 +1424,3 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H431 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -1405,0 +1429,3 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H432 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -1412,0 +1439,6 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H433 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -2821,0 +2854,12 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H434 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -2946,0 +2991,10 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H435 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -2953,0 +3008,7 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H436 | `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | `@@ -3050,0 +3112,8 @@ class VLLMv1EngineReplicaScheduler(BaseReplicaScheduler):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H437 | `frontier/scheduler/utils/memory_planner.py` | `@@ -4 +4,2 @@ from frontier.attention.memory import get_attention_runtime_kv_layout` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H438 | `frontier/scheduler/utils/memory_planner.py` | `@@ -18,0 +20 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H439 | `frontier/scheduler/utils/memory_planner.py` | `@@ -22,0 +25,3 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H440 | `frontier/scheduler/utils/memory_planner.py` | `@@ -97 +102,20 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H441 | `frontier/scheduler/utils/memory_planner.py` | `@@ -119 +143 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H442 | `frontier/scheduler/utils/memory_planner.py` | `@@ -130 +154,2 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H443 | `frontier/scheduler/utils/memory_planner.py` | `@@ -132 +157,34 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H444 | `frontier/scheduler/utils/memory_planner.py` | `@@ -133,0 +192,14 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H445 | `frontier/scheduler/utils/memory_planner.py` | `@@ -153 +225,3 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H446 | `frontier/scheduler/utils/memory_planner.py` | `@@ -178 +252,4 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H447 | `frontier/scheduler/utils/memory_planner.py` | `@@ -187,0 +265 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H448 | `frontier/scheduler/utils/memory_planner.py` | `@@ -194,0 +273,3 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H449 | `frontier/scheduler/utils/memory_planner.py` | `@@ -196 +277 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H450 | `frontier/scheduler/utils/memory_planner.py` | `@@ -210 +291,3 @@ class MemoryPlanner:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H451 | `frontier/simulator.py` | `@@ -66,0 +67,6 @@ class Simulator:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H452 | `frontier/simulator.py` | `@@ -182 +188,4 @@ class Simulator:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H453 | `frontier/simulator.py` | `@@ -186,0 +196,18 @@ class Simulator:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H454 | `frontier/simulator.py` | `@@ -196 +223 @@ class Simulator:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H455 | `frontier/simulator.py` | `@@ -197,0 +225,2 @@ class Simulator:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H456 | `frontier/training/__init__.py` | `@@ -16,0 +17 @@ from frontier.training.attention_trainer import AttentionTrainer` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H457 | `frontier/training/__init__.py` | `@@ -25,0 +27 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H458 | `frontier/training/__init__.py` | `@@ -29 +30,0 @@ __all__ = [` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H459 | `frontier/training/attention_trainer.py` | `@@ -383 +383 @@ class AttentionTrainer(BaseTrainer):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H460 | `frontier/training/base_trainer.py` | `@@ -301,0 +302 @@ class BaseTrainer(ABC):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H461 | `frontier/training/cli.py` | `@@ -20,0 +21 @@ from frontier.training.attention_trainer import AttentionTrainer, create_attenti` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H462 | `frontier/training/cli.py` | `@@ -331,0 +333,27 @@ Examples:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H463 | `frontier/training/cli.py` | `@@ -636,0 +665,31 @@ def train_attention(args):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H464 | `frontier/training/cli.py` | `@@ -654,0 +714,2 @@ def main():` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H465 | `frontier/training/gdn_trainer.py` | `@@ -0,0 +1,341 @@` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H466 | `frontier/types/device_sku_type.py` | `@@ -12,0 +13 @@ class DeviceSKUType(BaseIntEnum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H467 | `frontier/types/measurement_type.py` | `@@ -5,0 +6 @@ class MeasurementType(str, Enum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H468 | `frontier/types/node_sku_type.py` | `@@ -13,0 +14 @@ class NodeSKUType(BaseIntEnum):` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H469 | `frontier/utils/param_counter.py` | `@@ -375,0 +376,102 @@ class ParamCounter:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H470 | `frontier/utils/param_counter.py` | `@@ -397,10 +499 @@ class ParamCounter:` | NOT_YET_REVIEWED | Pending package/caller inspection |
| H471 | `frontier/utils/param_counter.py` | `@@ -449,0 +543,13 @@ class ParamCounter:` | NOT_YET_REVIEWED | Pending package/caller inspection |
