#!/bin/bash

pushd `dirname $0` > /dev/null
cur_dir=`pwd`
popd > /dev/null

shop_type=FSL
TMP_XML=resnet_${shop_type}_dist_predict.xml
APP_NAME=predict_${shop_type}_pic
MODEL_HDFS_PATH=viewfs://hadoop-meituan/ghnn01/user/hadoop-dpsr/huangyanchun/fsl/model/relation_network_keras.pkl
INPUT_HDFS_PATH=viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr_test.db/mainpic_fetch_ugcpics_needclassify_all
#INPUT_HDFS_PATH=viewfs://hadoop-meituan/nn01/warehouse/upload_table.db/manpic_fsl_native_hdfs_compare
OUTPUT_HDFS_PATH=viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/mainpic_fsl_predict/class=6_10
WORKER_SCRIPT=common_distributed_predict_resnet.py
MODEL_DEFINE_NAME=pointwise_basic_dnn_model.py
WORKERS=10


source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh hadoop-dpsr
source /opt/meituan/tensorflow-release/local_env.sh
export JAVA_HOME="/usr/local/java"
unset CLASSPATH


hadoop fs -chmod -R g+r $INPUT_HDFS_PATH
hadoop fs -chmod g+rw $OUTPUT_HDFS_PATH


sed -e "s@APP_NAME@${APP_NAME}@" $cur_dir/resnet_common_dist_predict.xml > $cur_dir/$TMP_XML
sed -i "s@MODEL_HDFS_PATH@${MODEL_HDFS_PATH}@" $cur_dir/$TMP_XML
sed -i "s@INPUT_HDFS_PATH@${INPUT_HDFS_PATH}@" $cur_dir/$TMP_XML
sed -i "s@OUTPUT_HDFS_PATH@${OUTPUT_HDFS_PATH}@" $cur_dir/$TMP_XML
sed -i "s@WORKER_SCRIPT@${WORKER_SCRIPT}@" $cur_dir/$TMP_XML
sed -i "s@MODEL_DEFINE_NAME@${MODEL_DEFINE_NAME}@" $cur_dir/$TMP_XML
sed -i "s@WORKERS@${WORKERS}@" $cur_dir/$TMP_XML


bash ${AFO_TF_HOME}/bin/tensorflow-submit.sh -conf $cur_dir/$TMP_XML -files $cur_dir/$WORKER_SCRIPT,$MODEL_DEFINE_NAME,$cur_dir/pointwise_softmax_dnn_model.py

sleep 5
chmod 777 $cur_dir/$TMP_XML