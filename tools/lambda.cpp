#include <string>
#include <iostream>
#include <caffe/caffe.hpp>

int main(int argc, char** argv) {
  if (argc != 3){
    std::cerr << "Usage: " << argv[0] << " MODEL WEIGHTS";
  }
  caffe::Net<float> net(argv[1], caffe::TEST);
  net.CopyTrainedLayersFrom(argv[2]);
  std::string line; 
  std::cout << "paused.";
  std::getline(std::cin, line);
  std::cout << "Done.";
}

