for i in {10..16}
do
 if [ $i -lt 10 ]
 then
  fname="test_00$i"
  fname+="aah.tif"
 else
  fname="test_0$i"
  fname+="aah.tif"
 fi
 python ./wsi/bin/probs_map.py ./sample_data/split_files/$fname ./ckpt/resnet18_crf.ckpt ./configs/resnet18_crf.json ./sample_data/mask_files_level8/test_0$i.npy ./sample_data/probmaps_level8/test_0$i.npy 
done
