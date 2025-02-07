#!/bin/bash

# apt install -y zlib1g zlib1g-dev libssl-dev libbz2-dev uuid-dev libsqlite3-dev liblzma-dev
# wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz
# tar xvfz Python-3.11.11.tgz
# cd Python-3.11.11/
# ./configure --enable-optimizations && make -j16 && sudo make altinstall

python3.11 -m venv ./pythonenv/

# fixed dependencies created with "pip freeze" in "requirements.txt"
#
# consider fetching these if newer dependencies create problems

