xhost local:$USER
docker run --gpus all -it --name=fzero_container --rm -p 5557:5557 -p 5558:5558 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fzero
