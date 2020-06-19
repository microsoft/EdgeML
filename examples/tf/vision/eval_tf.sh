#! /bin/bash

# MSR
IS_QVGA_MONO=1 python tf_keras_eval.py --model_arch RPool_Face_QVGA_monochrome --model weights/rpool_face_qvgamono_withscut_trainaugfacele48.pth --image_folder original --save_dir results-original $@
