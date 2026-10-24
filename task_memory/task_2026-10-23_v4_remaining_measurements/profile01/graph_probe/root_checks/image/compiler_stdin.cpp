#include "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/cupti_callbacks.h"
static_assert(sizeof(CUpti_ResourceData) == 24 && alignof(CUpti_ResourceData) == 8);
static_assert(__builtin_offsetof(CUpti_ResourceData, context) == 0 && __builtin_offsetof(CUpti_ResourceData, resourceHandle) == 8 && __builtin_offsetof(CUpti_ResourceData, resourceDescriptor) == 16);
static_assert(sizeof(((CUpti_ResourceData*)0)->resourceHandle) == 8 && __builtin_offsetof(CUpti_ResourceData, resourceHandle.stream) == 8);
static_assert(sizeof(CUpti_GraphData) == 56 && alignof(CUpti_GraphData) == 8 && __builtin_offsetof(CUpti_GraphData, node) == 16 && __builtin_offsetof(CUpti_GraphData, originalNode) == 24 && __builtin_offsetof(CUpti_GraphData, nodeType) == 32 && __builtin_offsetof(CUpti_GraphData, dependency) == 40 && __builtin_offsetof(CUpti_GraphData, graphExec) == 48);
