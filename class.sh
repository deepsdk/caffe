set -e

make all -j4

#./build/example/caffe test -iterations 1 -model "../model/lenet_id_data_layer.prototxt" -weights "../model/lenet_iter_10000.caffemodel" $@  #--minloglevel=1

MEAN="./data/ilsvrc12/imagenet_mean.binaryproto"
LABELS="./data/ilsvrc12/synset_words.txt"

DEPLOY="./models/bvlc_reference_caffenet/deploy.prototxt"
MODEL="./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@

echo ""

DEPLOY="./models/bvlc_googlenet/deploy.prototxt"
MODEL="./models/bvlc_googlenet/bvlc_googlenet.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@

echo ""

DEPLOY="./models/bvlc_alexnet/deploy.prototxt"
MODEL="./models/bvlc_alexnet/bvlc_alexnet.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@

echo ""

DEPLOY="./models/zoo_nin_imagenet/deploy.prototxt"
MODEL="./models/zoo_nin_imagenet/nin_imagenet.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@

echo ""

DEPLOY="./models/zoo_vgg_ilsvrc_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt"
MODEL="./models/zoo_vgg_ilsvrc_16_layers/VGG_ILSVRC_16_layers.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@

echo ""

DEPLOY="./models/zoo_vgg_ilsvrc_19_layers/VGG_ILSVRC_19_layers_deploy.prototxt"
MODEL="./models/zoo_vgg_ilsvrc_19_layers/VGG_ILSVRC_19_layers.caffemodel"
echo $MODEL
build/examples/cpp_classification/classification.bin $DEPLOY $MODEL $MEAN $LABELS $@
