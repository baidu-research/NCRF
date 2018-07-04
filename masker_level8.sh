for i in {1..16}
do
 if [ $i -lt 10 ]
 then
  fname="test_00$i"
  fname+="aah.tif"
 else
  fname="test_0$i"
  fname+="aah.tif"
 fi
 python ./wsi/bin/tissue_mask.py ./sample_data/split_files/$fname ./sample_data/mask_files_level8/test_0$i.npy --level 0
done
