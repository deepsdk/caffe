set -e

make all -j4
./build/tools/caffe test -iterations 1 -model "../model/lenet_id_data_layer.prototxt" -weights "../model/lenet_iter_10000.caffemodel" $@   # --minloglevel=1
