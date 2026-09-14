# PR31 hybrid fixture (test-only)

This directory contains a small eight-layer BF16 Qwen3.5-shaped model and
synthetic GDN profiling rows used by CPU unit/integration tests.  The schedule
is `GDN, GDN, GDN, FullAttn, GDN, GDN, GDN, FullAttn`.

The tensors, timings, and metadata are synthetic.  They do not represent
MI355X hardware measurements, checkpoint-exact weights, ROCm execution, or
benchmark/ground-truth parity.  Production profiling and model-manager input
paths must not discover this fixture automatically.
