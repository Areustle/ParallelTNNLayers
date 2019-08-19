#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#include "../cp4Conv2d.cuh"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<unsigned N,
         unsigned C,
         unsigned H,
         unsigned W,
         unsigned pad,
         unsigned fK,
         unsigned fH,
         unsigned fW,
         unsigned fRank>
void profile(streambuf* output_buffer) {
  float us = CP::run_convolution<N, C, H, W, pad, fK, fH, fW, fRank>(47);

  ostream results(output_buffer);
  results << N << "," << C << "," << H << "," << W << "," << pad << "," << fK
          << "," << fH << "," << fW << "," << fRank << ", " << us << endl;
}

template<unsigned rank> void profile_helper(streambuf* output_buffer) {

  profile_helper<rank - 1>(output_buffer);

  profile<1, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<2, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<4, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<8, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<16, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<32, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<64, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
  profile<128, 3, 512, 512, 1, 1, 3, 3, rank>(output_buffer);
}

template<> void profile_helper<0>(streambuf* output_buffer) {}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  streambuf* output_buffer = std::cout.rdbuf();
  ofstream   of;
  int        device = 0;

  switch (argc) {
    case 3: device = atoi(argv[2]);
    case 2: of.open(argv[1]); output_buffer = of.rdbuf();
    case 1: break;
    default:
      cerr << "USAGE: BenchCP4 "
              " [Results_file] "
              " [device_number]"
           << endl;
      return 1;
  }

  cudaSetDevice(device);

  ostream results(output_buffer);
  results << "N,C,H,W,pad,fK,fH,fW,fRank,us" << endl;

  profile_helper<16>(output_buffer);
}
