# DCC_Driving_Anomaly_Detection-
Code for Paper: “Dual-stream Model Combining Spatio-Temporal and Appearance Features for Anomaly Driving Action Detection”

The DAD dataset used in this work：https://pan.baidu.com/s/1qVAXj82gkw82hbyb3kDoCg code：8hgs 
（There may be an error during unzip. Please record the corresponding error image, copy and rename the adjacent image to replace it. There are about 4 wrong images）

The pre-training weights used in the code：https://pan.baidu.com/s/1u-u2SKDi43NSzTR1fyk0Xw code：qgda 
Download the weights and add them to the 'premodels' folder

The environment configuration is in requirement.txt and may need to be supplemented with some packages
use python3.8, torch 1.12.1 torchvision 0.13.1 torchaudio 0.12.1+cu113

After preparations are complete, change the 'root_path' in main ArgumentParser and replace the address of the directory where the DAD data set resides

Modify the 'view' to train 4 different views

'batch_size' can be set according to your own devices, the default batch_size is the best parameter configuration

'device' can be configured according to their own Settings, support multi-gpu training

Other main ArgumentParser do not need to be adjusted

# Train model
you can use 'python main.py' to train

# Test model
I've uploaded the calculated score, so you don't need weights or datasets to test
just use 'python main.py --mode test' 

# Acknowledgement
Thanks for open source of 'https://github.com/okankop/Driver-Anomaly-Detection', my code to solve the problems of memory in the source code
