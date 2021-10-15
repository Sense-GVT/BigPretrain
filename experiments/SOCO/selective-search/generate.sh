$patch=5000
for low in $(seq 0 $patch 1281168);
do
    high=$(($low + $patch))
    nohup spring.submit run --job-name ${low} "python ssearch_imagenet.py --start ${low} --end ${high}" >> ./log/log${low}.txt 2>&1 &
done
