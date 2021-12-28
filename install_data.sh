#!/bin/bash -v

sudo pip3 install --force-reinstall opencv-python==3.4.10.35 tqdm websockets
sudo pip3 install --force-reinstall gdown

gdown https://drive.google.com/u/0/uc?id=13eVblwuqgec1qQkXRqoA_Kxj4hDWKbFq&export=download

while :
do
if [ -f ./tb_data.tar.gz ]; then
    break
fi
sleep 1
done

tar -vxf ./tb_data.tar.gz
