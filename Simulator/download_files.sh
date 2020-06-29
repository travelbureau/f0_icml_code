#/bin/bash

./gdown.pl 'https://drive.google.com/open?id=1hW3X9IGq7fu_1aw6joTI_WU6_gai7Pwy' lut_inuse.npz
mv lut_inuse.npz ./python/mpc

./gdown.pl 'https://drive.google.com/open?id=11NVCYd0n5BnVfbhrT-hoiVA2-dEtQDOl' flow_weights.zip
unzip flow_weights.zip
rm flow_weights.zip
rm -rf __MACOSX
mv flow_weights ./python

./gdown.pl 'https://drive.google.com/open?id=15R3SSn3Utn05HDMXY2MjvcedWR3C1AKR' map1_range.msgpack
./gdown.pl 'https://drive.google.com/open?id=1vPFcLqzhbTZdpxHOU-o-3GnWe1atfshG' map1_speed.msgpack
mv map1_range.msgpack ./python
mv map1_speed.msgpack ./python
