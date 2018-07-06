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
 python ./wsi/bin/probs_map.py ./data/$fname ./ckpt/resnet18_crf.ckpt ./configs/resnet18_crf.json ./data/mask_files/test_0$i.npy ./data/probmaps/test_0$i.npy
done
