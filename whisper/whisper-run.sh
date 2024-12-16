#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <file-path>"
    exit 1
fi

whisper --model small --task transcribe --language vi "$1"