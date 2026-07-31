#!/bin/bash
rosdep update 
sudo apt-get update 
sudo chown $(id -u):$(id -g) build install log
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y