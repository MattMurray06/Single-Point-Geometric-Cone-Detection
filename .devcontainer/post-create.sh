#!/bin/bash
rosdep update 
sudo apt-get update 
git submodule update --init --recursive
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y