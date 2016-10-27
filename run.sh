set -e

make all -j4
#./build/tools/caffe test -iterations 1 -model "../model/lenet_id_data_layer.prototxt" -weights "../model/lenet_iter_10000.caffemodel" $@  #--minloglevel=1
#./build/tools/lambda.bin "../model/lenet_id_data_layer.prototxt" "../model/lenet_iter_10000.caffemodel"



MODEL="./models/bvlc_reference_caffenet/deploy.prototxt"
WEIGHTS="./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#MODEL="./models/bvlc_googlenet/deploy.prototxt"
#WEIGHTS="./models/bvlc_googlenet/bvlc_googlenet.caffemodel"
#MODEL="./models/bvlc_alexnet/deploy.prototxt"
#WEIGHTS="./models/bvlc_alexnet/bvlc_alexnet.caffemodel"
./build/tools/lambda.bin $MODEL $WEIGHTS
