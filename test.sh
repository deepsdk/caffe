set -e

source make.sh
make test -j4
./build/test/test_id_data_layer.testbin 
