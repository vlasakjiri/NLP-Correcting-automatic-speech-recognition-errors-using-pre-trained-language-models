# wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# tar xJfv ffmpeg-release-amd64-static.tar.xz


TOPDIR=/storage/brno12-cerit/home/xvlasa15
REPODIR=$TOPDIR/knn-whisper
DATASETDIR=$TOPDIR/cache


rsync -rh --info=progress2 $REPODIR $DATASETDIR "$SCRATCHDIR"



cd $SCRATCHDIR/knn-whisper
pip install -r requirements.txt

# -- do your amazing stuf here
python ./src/fine_tune_whisper.py