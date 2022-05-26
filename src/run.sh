#!/bin/bash


#if [ ! -d models ]; then
#     mkdir models
#     cd models
#
#     filename="FlowNet2_checkpoint.pth.tar"
#     fileid="1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da"
#     # fileid="157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8"
#
#
#     wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='${fileid} -O- \
#          | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
#
#     wget --load-cookies cookies.txt -O ${filename} \
#          'https://docs.google.com/uc?export=download&id='${fileid}'&confirm='$(<confirm.txt)
#
#     cd ..
#fi
python3 -m venv "test-env"
source "test-env/bin/activate"
python -m pip install --upgrade pip
python -m pip install opencv-python

python3 detect_morph_lines.py /csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/resources/side_hearts.jpg
