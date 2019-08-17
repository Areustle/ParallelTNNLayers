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
  if (argc != 4) {
    cerr << "USAGE: BenchCP4 "
            " Tensor_file "
            " Results_file "
            " device_number"
         << endl;
    return 1;
  }

  ifstream tensors(argv[1]);
  ofstream results(argv[2]);
  int      device(atoi(argv[3]));

  results << "N,C,H,W,pad,fK,fH,fW,fRank,ns" << endl;


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
    float ns = NV::run_convolution(p, 47);
    results << p.N << "," << p.C << "," << p.H << "," << p.W << "," << p.pad
            << "," << p.fK << "," << p.fH << "," << p.fW << "," << ns << endl;
  }
}
