#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/data/miniconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

/data/miniconda3/bin/bert-serving-start -pooling_layer -1 -model_dir /data/ceph/yuncongli/word-vector/uncased_L-12_H-768_A-12 -max_seq_len=NONE -num_worker=4 -port=8880 -pooling_strategy=NONE -cpu -show_tokens_to_client > bert_service.log 2>&1 &

for i in `seq 0 4`
do
	${python} $@  --use_bert True --emb_file /data/ceph/yuncongli/word-vector/glove.840B.300d.txt --current_run $i
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
