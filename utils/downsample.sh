echo "Will convert images in all subdirectoies of $1";
for D in "$1"*/; do # for all directories in the provided path
	echo "Will convert images in $D"; # current directory
	if [ ! -d "$D/downsampled" ]; then # if the subdirectory does not already exist: create it
		mkdir "$D/downsampled"; # create location for downsampled images
	fi
	if [ "$(ls -A $D/downsampled)" ]; then # if it already contains images, do nothing to not delete anyting accidentaly
		echo "$D/downsampled is not empty, will do nothing";
	else # else: start converting images
		for filepath in "$D"*.png; do # for every file 
			filename=`basename "$filepath"` # strip directory prefix
			echo "Converting $filename";
			ffmpeg -i "$filepath" -vf scale=-1:120 "$D/downsampled/${filename%.png}_-1x120.png"; # downsample
		done
	fi
done
