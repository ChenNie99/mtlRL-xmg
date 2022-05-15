#!/bin/bash
read -p "Please Enter You file: " NAME
echo "Your FileName Is: $NAME"
cp $NAME /home/abcRL2.0-4-24/results
cd /home/abcRL2.0-4-24/results
git add $NAME
git commit -m "add $NAME"
git reset --hard
git pull
git push

