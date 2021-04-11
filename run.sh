#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/data/miniconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

for i in `seq 0 4`
do
	${python} $@ --emb_file glove.840B.300d.txt --current_run $i
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
