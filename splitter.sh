for i in {1..16}
do
 if [ $i -lt 10 ]
 then
  tiffsplit ./sample_data/test_00$i.tif ./sample_data/split_files/test_00$i
 else
  tiffsplit ./sample_data/test_0$i.tif ./sample_data/split_files/test_0$i
 fi
done
