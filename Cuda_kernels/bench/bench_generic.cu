#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../cp4Conv2d.cuh"

using namespace std;

float profile_cuda_conv2d_cp4_gpu(string exe, string line, string device) {

  string cmd = "nv-nsight-cu-cli "
               "--metrics gpu__time_duration.avg "
               "--csv "
               "--unit base "
               "--device "
               + device + " " + exe + " " + line + " " + device
               + " | tail -1"
                 " | rev"
                 " | cut -d '\"' -f 1-2"
                 " | cut -c 2-"
                 " | rev"
                 " | sed 's/,//g'";

  array<char, 128> buffer;
  string           result;

  unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) { throw runtime_error("popen() failed!"); }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    result += buffer.data();

  return stof(result);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    cerr << "USAGE: Benchmark Executable Tensor_file Results_file device_number"
         << argc << endl;
    return 1;
  } else {

    string   exe(argv[1]);
    ifstream tensors(argv[2]);
    ofstream results(argv[3]);
    string   device(argv[4]);

    results << "N,C,H,W,pad,fK,fH,fW,fRank,ns" << endl;

    if (!tensors.is_open()) {
      cerr << "Couldn't open tensors file.\n";
    } else {
      string line;
      while (getline(tensors, line)) {

        if (line[0] == '#' || line.empty()) continue;

        stringstream line_sm(line);
        float        ns = profile_cuda_conv2d_cp4_gpu(exe, line, device);
        unsigned     N, H, W, C, pad, fK, fH, fW, fRank;
        line_sm >> N >> C >> H >> W >> pad >> fK >> fH >> fW >> fRank;

        results << N << "," << C << "," << H << "," << W << "," << pad << ","
                << fK << "," << fH << "," << fW << "," << fRank << "," << ns
                << endl;
      }
    }
  }
}
