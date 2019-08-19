# Input      |  Filter (actually fK C fH fW, but C taken from Input)
# Here the first 4 entries are the NCHW shape of the Input tensor, followed by
# the number of cells to zero-pad (along the H&W dimensions). Then the filter
# tensor shape where channel depth is implied by C and the tensor rank.
# 
# N: Input data batch size
# C: Shared channel depth of input and filter tensors
# H: Input tensor height
# W: Input tensor width
# pad: depth of cells to zero-pad along HW border of input
# 
# fK: Output channel depth after convolution.
# fH: filter height of convolution kernel
# fW: filter width of convolution kernel
# fRank: The rank of the tensor decomposition.
#
#N C H W pad | fK fH fW fRank

# scale RGB images

# rank 1
1 3   32   32 1    1 3 3 1
1 3   64   64 1    1 3 3 1
1 3  128  128 1    1 3 3 1
1 3  256  256 1    1 3 3 1
1 3  512  512 1    1 3 3 1
1 3 1024 1024 1    1 3 3 1
1 3 2048 2048 1    1 3 3 1
1 3 4096 4096 1    1 3 3 1

# rank 2
1 3   32   32 1    1 3 3 2
1 3   64   64 1    1 3 3 2
1 3  128  128 1    1 3 3 2
1 3  256  256 1    1 3 3 2
1 3  512  512 1    1 3 3 2
1 3 1024 1024 1    1 3 3 2
1 3 2048 2048 1    1 3 3 2
1 3 4096 4096 1    1 3 3 2

# rank 4
1 3   32   32 1    1 3 3 4
1 3   64   64 1    1 3 3 4
1 3  128  128 1    1 3 3 4
1 3  256  256 1    1 3 3 4
1 3  512  512 1    1 3 3 4
1 3 1024 1024 1    1 3 3 4
1 3 2048 2048 1    1 3 3 4
1 3 4096 4096 1    1 3 3 4

# rank 8
1 3   32   32 1    1 3 3 8
1 3   64   64 1    1 3 3 8
1 3  128  128 1    1 3 3 8
1 3  256  256 1    1 3 3 8
1 3  512  512 1    1 3 3 8
1 3 1024 1024 1    1 3 3 8
1 3 2048 2048 1    1 3 3 8
1 3 4096 4096 1    1 3 3 8

# rank 16
1 3   32   32 1    1 3 3 16
1 3   64   64 1    1 3 3 16
1 3  128  128 1    1 3 3 16
1 3  256  256 1    1 3 3 16
1 3  512  512 1    1 3 3 16
1 3 1024 1024 1    1 3 3 16
1 3 2048 2048 1    1 3 3 16
1 3 4096 4096 1    1 3 3 16
