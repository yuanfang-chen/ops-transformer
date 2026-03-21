mkdir -p build; cd build
cmake ..; make -j
python3 ../fia_test.py