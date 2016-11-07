set -e

make all -j4

#./build/example/caffe test -iterations 1 -model "../model/lenet_id_data_layer.prototxt" -weights "../model/lenet_iter_10000.caffemodel" $@  #--minloglevel=1

MEAN="./data/ilsvrc12/imagenet_mean.binaryproto"
LABEL="./data/ilsvrc12/synset_words.txt"

#MODEL="./models/bvlc_reference_caffenet/deploy.prototxt"
#WEIGHTS="./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@
#
#echo ""
#
#MODEL="./models/bvlc_googlenet/deploy.prototxt"
#WEIGHTS="./models/bvlc_googlenet/bvlc_googlenet.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@
#
#echo ""
#
#MODEL="./models/bvlc_alexnet/deploy.prototxt"
#WEIGHTS="./models/bvlc_alexnet/bvlc_alexnet.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@
#
#echo ""
#
#MODEL="./models/zoo_nin_imagenet/deploy.prototxt"
#WEIGHTS="./models/zoo_nin_imagenet/nin_imagenet.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

#echo ""
#
#MODEL="./models/zoo_vgg_ilsvrc_16_layers/deploy.prototxt"
#WEIGHTS="./models/zoo_vgg_ilsvrc_16_layers/VGG_ILSVRC_16_layers.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

#echo ""
#
#MODEL="./models/zoo_vgg_ilsvrc_19_layers/deploy.prototxt"
#WEIGHTS="./models/zoo_vgg_ilsvrc_19_layers/VGG_ILSVRC_19_layers.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

#echo ""
#
#MODEL="./models/zoo_vgg_ilsvrc_16_layers/deploy_1.prototxt"
#WEIGHTS="./models/zoo_vgg_ilsvrc_16_layers/VGG_ILSVRC_16_layers_1.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

#echo ""
#
#LABEL="./models/zoo_googlenet_places205/categoryIndex_places205.csv"
#MODEL="./models/zoo_googlenet_places205/deploy_places205.protxt"
#WEIGHTS="./models/zoo_googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@
#
#echo ""
#
#LABEL="./models/zoo_googlenet_places205/categoryIndex_places205.csv"
#MODEL="./models/zoo_places_cnds_model/deploy.prototxt"
#WEIGHTS="./models/zoo_places_cnds_model/8conv3fc_DSN.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

#echo ""
#
#LABEL="./models/zoo_googlenet_sos/counts.txt"
#MODEL="./models/zoo_googlenet_sos/deploy.prototxt"
#WEIGHTS="./models/zoo_googlenet_sos/GoogleNet_SOS.caffemodel"
#echo $WEIGHTS
#build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -label $LABEL -mean 104,117,123 $@


echo ""

MEAN="./models/zoo_cnn_agegender/mean.binaryproto"
LABEL="./models/zoo_cnn_agegender/age.txt"
MODEL="./models/zoo_cnn_agegender/deploy_age.prototxt"
WEIGHTS="./models/zoo_cnn_agegender/age_net.caffemodel"
echo $WEIGHTS
build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@

echo ""

MEAN="./models/zoo_cnn_agegender/mean.binaryproto"
LABEL="./models/zoo_cnn_agegender/gender.txt"
MODEL="./models/zoo_cnn_agegender/deploy_gender.prototxt"
WEIGHTS="./models/zoo_cnn_agegender/gender_net.caffemodel"
echo $WEIGHTS
build/examples/cpp_classification/classification.bin -model $MODEL -weights $WEIGHTS -meanfile $MEAN -label $LABEL $@


