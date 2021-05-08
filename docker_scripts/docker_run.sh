#!/bin/bash

echo ""
echo "Running drake_workspace container"
echo ""

sudo docker run -it \
    -v /home/agrobenj/drake_workspace:/home/$USER/workspace \
    -p 2400:22 \
    --name drake_workspace \
    drake_workspace \
    /bin/bash
