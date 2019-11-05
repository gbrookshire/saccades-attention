#!/bin/bash

# Convert eyelink EDF files to ASCII

for fullfile in ../data/eyelink/*; do
    
    pathname=$(dirname -- "$fullfile")
    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"

    # Skip directories
    if [ -d $fullfile ]
    then
        continue

    # Skip files with an ext
    elif [ "$filename" != "$extension" ] 
    then
        continue
    fi

    # Append `.edf` to the end of the eyelink files
    newpathname="${pathname}/ascii/"
    newfilename="${newpathname}${filename}.edf" 

    cp $fullfile $newfilename # Make a temporary file
    edf2asc $newfilename # Convert from EDF to ASCII
    rm $newfilename # Remove remporary file
    
done
