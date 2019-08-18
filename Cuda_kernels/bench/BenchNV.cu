#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "../NVConv2d.cuh"

using namespace std;

vector<tensor_shape> get_unique_ordered_shapes(vector<tensor_shape> input) {

  vector<tensor_shape> output;
  set<tensor_shape>    seen;

  for (auto& shape : input) {
    if (seen.count(shape) == 0) output.push_back(shape);
    seen.insert(shape);
  }

  return output;
}

int main(int argc, char** argv) {

  ifstream   tensors(argv[1]);
  streambuf* output_buffer = std::cout.rdbuf();
  ofstream   of;
  int        device = 0;

  switch (argc) {
    case 4: device = atoi(argv[3]);
    case 3: of.open(argv[2]); output_buffer = of.rdbuf();
    case 2: break;
    default:
      cerr << "USAGE: BenchNV "
              " Tensor_file "
              " [Results_file] "
              " [device_number]"
           << endl;
      return 1;
  }

  ostream results(output_buffer);
  results << "N,C,H,W,pad,fK,fH,fW,us" << endl;


  if (!tensors.is_open()) {
    cerr << "Couldn't open tensors file.\n";
    return 1;
  }

  vector<tensor_shape> shapes;

  string line;

  while (getline(tensors, line)) {

    if (line[0] == '#' || line.empty()) continue;

    stringstream line_sm(line);
    unsigned     N, H, W, C, pad, fK, fH, fW, fRank;
    line_sm >> N >> C >> H >> W >> pad >> fK >> fH >> fW >> fRank;

    tensor_shape params;
    params.N     = N;
    params.C     = C;
    params.H     = H;
    params.W     = W;
    params.pad   = pad;
    params.fRank = 0;
    params.fK    = fK;
    params.fH    = fH;
    params.fW    = fW;

    shapes.push_back(params);
  }

  shapes = get_unique_ordered_shapes(shapes);

  cudaSetDevice(device);

  for (auto& p : shapes) {
    float us = NV::run_convolution(p, 47);
    results << p.N << "," << p.C << "," << p.H << "," << p.W << "," << p.pad
            << "," << p.fK << "," << p.fH << "," << p.fW << "," << us << endl;
  }
}
