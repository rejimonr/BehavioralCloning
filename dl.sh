#!/bin/bash
fileid="1EJPXwhQuvbfXboRj02AWBVKvZEtK7xTf"
filename="/opt/carnd_p3/own/driving_log.csv"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}