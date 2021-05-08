To convert the still frames into a video, extract the `.tar` file and run: 
```ffmpeg -r 60 -i %07d.png \
     -vcodec libx264 \
     -preset slow \
     -crf 18 \
     output.mp4```
