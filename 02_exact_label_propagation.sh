#!/bin/bash

# To run: ./02_label_propagation.sh file_of_paths_and_labels
# Each line of file_of_paths_and_labels has follwing format: /da1_data/icputrd/arf/mean.js/public/anau_img3/000/00000122.08.JPG,0 (image_path, label)
# This is exact date propagation (i.e., given a labeled image, other images of same donor with same date, will receive the same label as the labeled image)

out_file="data/3_classes/propagated_0_0/stages.csv.20221114.3_classes.propagated"
#out_file="data/4_classes/propagated_0_0/stages.csv.20221114.4_classes.propagated"

if [ -f "$out_file" ] ; then
    rm "$out_file"
fi

while read line; do
    day=$(echo $line | cut -d '/' -f 9 | cut -d '.' -f 1)
    #echo $day
    label=$(echo $line | cut -d ',' -f 2)
    #echo $label

    # propagate label
    if grep -q "$day" data/clusters.csv.head; then
        echo $line":"
        grep "$day" data/clusters.csv.head | while read -r line2; do
            img=$(echo $line2 | cut -d ',' -f 1)
            img2="/da1_data/icputrd/arf/mean.js/public"$img","$label
            echo "-->" $img2
            echo $img2 >> $out_file
        done
        echo ""
    fi
done < $1

awk '{if(!seen[$0]++)print $0}' $1 $out_file > $out_file".merged"
