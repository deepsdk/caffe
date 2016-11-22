#!/bin/bash

#scp ./build/lib/libdlib.so.19.2.99 lambda:~/package/lib
scp ./build/tools/classification_tcnn.bin lambda:~/package/bin
