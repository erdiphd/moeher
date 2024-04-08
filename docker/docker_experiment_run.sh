#!/bin/bash

docker-compose run --rm -d -e mujoco_env=FetchPush-v1 -e log_tag=log/train1 -e n_epochs=900 cher
docker-compose run --rm -d -e mujoco_env=FetchPush-v1 -e log_tag=log/train2 -e n_epochs=900 cher
docker-compose run --rm -d -e mujoco_env=FetchPush-v1 -e log_tag=log/train3 -e n_epochs=900 cher
docker-compose run --rm -d -e mujoco_env=FetchPush-v1 -e log_tag=log/train4 -e n_epochs=900 cher
docker-compose run --rm    -e mujoco_env=FetchPush-v1 -e log_tag=log/train7 -e n_epochs=900 cher

