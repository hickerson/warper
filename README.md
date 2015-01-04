warper
======
Modify the darkmatterwarper.py to point to a dir you want to store frames. 
These should probably all be in their own directory labeled "frames".

then run 

    python darkmatterwarper.py
    
then run 

    cd <dir with your below your frames dir>
    avconv -i frames/frame%06d.png -r 24 -vcodec mpeg4 -vb 20M <your movie name>.mp4
    
note that you need avconv for this. This use to be called ffmpeg. 
