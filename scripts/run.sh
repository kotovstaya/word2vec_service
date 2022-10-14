#! /bin/bash

#./clone_repo.sh
#cd ./../jb && python raw.py

#cd ./../jb && python train_frombox.py from-box
#cd ./../jb && python train_vanilla.py coocurance-vanilla --count-in-line=512 --batch-size=1 --emb-dim=150 --epochs=2
cd ./../jb && python train_ns.py coocurance-ns --count-in-line=256 --batch-size=1 --emb-dim=150 --epochs=1

#cd ./../jb && python visualization.py

#cd ./../jb && python train_vanilla.py vanilla --window-size=5 --count-in-line=8 --batch-size=512 --emb-dim=150 --epochs=5
#cd ./../jb && python train_ns.py simple-ns --window-size=5 --count-in-line=8 --batch-size=512 --emb-dim=150 --epochs=5
