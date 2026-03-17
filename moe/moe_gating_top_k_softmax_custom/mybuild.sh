#!/bin/bash

source /usr//local/Ascend8.5.0/ascend-toolkit/latest/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

cur=$(pwd)

mode=$1

if [ "X$mode" = "X" ];then
	mode="build"
fi


case "$mode" in
run )
	
	echo "#### Begin Test ####"
	rm -rf ./input/*
	rm -rf ./output/*
	rm -rf ./golden/* vendors
	mkdir input
	./MoeGatingTopkSoftmax/build_out/custom_opp_ubuntu_aarch64.run --install-path=$(pwd)/
	source $(pwd)/vendors/customize/bin/set_env.bash
	python3 gen_data.py
	python3 compare_and_expert.py --input "./input/input0.pth"
	;;
* )
	echo "#### Begin Build ####"
	cd MoeGatingTopkSoftmax
	chmod -R 777 .
	bash build.sh
	cd $cur
	;;
esac