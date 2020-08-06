cd ./utils/RoIAlign;
sh build.sh;
cd ../sdf_layer/;
sh build.sh;
cd ../posecnn_layer/;
python setup.py build develop;
cd ../../utils;
python setup.py build_ext --inplace

