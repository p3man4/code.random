#!/bin/bash


root="/home/junwon/smt-data/images/*"

for dir in $(find $root -type d)
do
    numFiles=$(ls $dir | wc -l)
    name=$(echo $dir| cut -d"/" -f7)
    echo $name":"$numFiles >> log2.txt
done
