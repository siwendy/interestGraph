#!/bin/sh
/usr/bin/hadoop/software/hadoop/bin/hadoop fs -rmr /home/xitong/tf-test/outputTest
/usr/bin/hadoop/software/hbox/bin/hbox-submit \
   --app-type "tensorflow" \
   --driver-memory 1024 \
   --driver-cores 1 \
   --files tfTestDemo.py,dataDeal.py \
   --worker-memory 8192 \
   --worker-num 2 \
   --worker-cores 1 \
   --worker-gpus 1 \
   --ps-memory 1024 \
   --ps-num 2 \
   --ps-cores 1 \
   --hbox-cmd "python tfTestDemo.py --data_path=./data --save_path=./model --log_dir=./eventLog --training_epochs=10" \
   --board-enable true \
   --output /home/xitong/tf-test/outputTest#model \
   --input /home/xitong/tf-test/inputfile#data \
   --priority VERY_LOW \
   --conf tf.driver.memory=2048 \
   --app-name "tfdemo" 
echo "view result"
hadoop fs -ls /home/xitong/tf-test/outputTest

