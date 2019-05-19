#! /bin/bash

# Run the correctnes tests.
python test_conv2d_cp.py
python test_dense_cp.py
python test_conv2d_rcp.py

# Run the performance Benchmarks.
python benchmark_conv2d_cp.py
python benchmark_dense_cp.py
python benchmark_conv2d_rcp.py
