#!/bin/bash

source /home/gerard_rosell_cardus/SegmentationModel_PP_AI/global

me=$(echo "$0" | sed -e 's/.sh//g' | sed -e 's,.*/,,')
LOG=$LOG_DIR
LOG+=$me.log

cd $DIR

source venv/bin/activate

print_log "Starting training with arguments: $*"
python src/main.py "$@" 2>&1 | ts "[%Y-%m-%d %H:%M:%S] - $me:" >> $LOG