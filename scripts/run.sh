#! /bin/bash

./clone_repo.sh
cd ./../jb && python raw.py
cd ./../jb && python train.py
cd ./../jb && python visualization.py
