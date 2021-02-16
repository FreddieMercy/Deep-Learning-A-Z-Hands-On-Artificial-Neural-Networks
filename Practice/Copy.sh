#!/bin/sh

rm -rf Tensorflow\ Colab\ Practice > /dev/null 2>&1
cp -rf /Users/freddie/Google\ Drive/Colab\ Notebooks Tensorflow\ Colab\ Practice

git add **

TimeStamp="Update files at "+$(date)

git commit -m "add Tensorflow Colab Practice ($TimeStamp)"

git push