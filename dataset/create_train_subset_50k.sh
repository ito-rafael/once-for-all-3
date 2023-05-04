#!/usr/bin/env bash

SAMPLES='50'

dir_number=0
for DIR in $(ls train)
do
    echo $dir_number
    counter=0
    for IMAGE in $(ls train/$DIR)
    do
	# check if reached number of images
	if [[ "$counter" == $SAMPLES ]]
	then
	    break
	fi
        cp train/$DIR/$IMAGE train_subset_50k/$DIR/
	((counter++))
    done
    ((dir_number++))
done
