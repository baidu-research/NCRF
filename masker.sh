for i in {1..4}
do
 if [ $i -lt 10 ]
 then
  fname="test_00$i.tif"
  #fname+="aaf.tif"
 else
  fname="test_0$i.tif"
  #fname+="aaf.tif"
 fi
 python ./wsi/bin/tissue_mask.py ./data/$fname ./data/mask_files/test_0$i.npy
done
