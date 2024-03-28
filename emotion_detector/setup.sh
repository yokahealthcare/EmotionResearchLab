#!/bin/sh
pip install ultralytics
pip install tensorflow==2.15.0
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install numpy==1.23.5

yolo export model=yolov8n-face.pt format=engine device=0