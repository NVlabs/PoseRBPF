cd ./utils/RoIAlign;
sh build.sh;
cd ../sdf_layer/;
sh build.sh;
cd ../../utils;
python setup.py build_ext --inplace

