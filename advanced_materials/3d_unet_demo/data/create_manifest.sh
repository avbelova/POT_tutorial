cd images

MANIFEST_FILE=../manifest.csv
echo "image,label" > $MANIFEST_FILE

for f in `find . -name 'BRATS*.nii.gz'`
do
  name=`basename $f`
  echo images/$name,labels/$name >>  $MANIFEST_FILE

done
