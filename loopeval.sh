#!/bin/bash

for i in {20..30}
do
	echo "--------------------start eval model-$i.pt--------------------"
	rm -f ./checkpoint/model-last.pt
	cp -f ./checkpoint/model-$i.pt ./checkpoint/model-last.pt
	python3 ./eval.py --cuda True
done
