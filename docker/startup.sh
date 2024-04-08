#!/bin/bash

# This will run vnc server for gui
# sudo /usr/bin/supervisord -c /etc/supervisor/supervisord.conf  > /dev/null 2>&1 &


# /home/user/.conda/envs/hgg_poet/bin/python home/user/HGG/train.py
# /usr/bin/bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
source /home/user/conda/bin/activate her_ea
cd home/user/her_ea
sudo chown -R user:user /home/user/her_ea/
pip install -e /home/user/her_ea/gym
python train.py

