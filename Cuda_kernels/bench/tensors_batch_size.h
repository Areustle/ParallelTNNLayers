#include <array>
#include <tuple>
/* # Input      |  Filter (actually fK C fH fW, but C taken from Input) */
/* # Here the first 4 entries are the NCHW shape of the Input tensor, followed
 * by */
/* # the number of cells to zero-pad (along the H&W dimensions). Then the filter
 */
/* # tensor shape where channel depth is implied by C and the tensor rank. */
/* # */
/* # N: Input data batch size */
/* # C: Shared channel depth of input and filter tensors */
/* # H: Input tensor height */
/* # W: Input tensor width */
/* # pad: depth of cells to zero-pad along HW border of input */
/* # */
/* # fK: Output channel depth after convolution. */
/* # fH: filter height of convolution kernel */
/* # fW: filter width of convolution kernel */
/* # fRank: The rank of the tensor decomposition. */
/* # */
/* #N C H W pad | fK fH fW fRank */

/* # scale batch size */

/* # rank 1 */

constexpr std::array<std::tuple<int, int, int, int, int, int, int, int, int>,
                     40>
    batches{
      // rank 1
      std::make_tuple(1, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(2, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(4, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(8, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(16, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(32, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(64, 3, 512, 512, 1, 1, 3, 3, 1),
      std::make_tuple(128, 3, 512, 512, 1, 1, 3, 3, 1),

      // rank 2
      std::make_tuple(1, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(2, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(4, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(8, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(16, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(32, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(64, 3, 512, 512, 1, 1, 3, 3, 2),
      std::make_tuple(128, 3, 512, 512, 1, 1, 3, 3, 2),

      // rank 4
      std::make_tuple(1, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(2, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(4, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(8, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(16, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(32, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(64, 3, 512, 512, 1, 1, 3, 3, 4),
      std::make_tuple(128, 3, 512, 512, 1, 1, 3, 3, 4),

      // rank 8
      std::make_tuple(1, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(2, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(4, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(8, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(16, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(32, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(64, 3, 512, 512, 1, 1, 3, 3, 8),
      std::make_tuple(128, 3, 512, 512, 1, 1, 3, 3, 8),

      // rank 16
      std::make_tuple(1, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(2, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(4, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(8, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(16, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(32, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(64, 3, 512, 512, 1, 1, 3, 3, 16),
      std::make_tuple(128, 3, 512, 512, 1, 1, 3, 3, 16),
    };
