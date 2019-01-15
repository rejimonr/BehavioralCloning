#!/bin/bash
fileid="1poxK3mVgrGQCWxMtCJJEfB0ZeXB1rSZB"
filename="/opt/carnd_p3/own/IMG.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}