"""Selected SGLang/AITER experimental primitives.

The package deliberately exports no runtime objects. Importing it is safe on a
CPU-only host; SGLang, AITER, and torch CUDA modules are imported only by the
builder functions that execute a GPU primitive.
"""

__all__ = ()
