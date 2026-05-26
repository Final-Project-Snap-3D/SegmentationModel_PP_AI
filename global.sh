#!/bin/bash

LOG_DIR="/home/gerard_rosell_cardus/logs/"
DIR="/home/gerard_rosell_cardus/SegmentationModel_PP_AI"

function print_log() {
    me=$(echo "$0" | sed -e 's,.*/,,')
    echo -e "$@" | ts "[%Y-%m-%d %H:%M:%S] - $me:" >> "$LOG"
    echo -e "$@" | ts "[%Y-%m-%d %H:%M:%S] - $me:"
}

exit 0