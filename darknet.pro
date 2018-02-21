QT += core
QT -= gui

CONFIG += c++11

TARGET = darknet
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

OPTIONS = GPU CUDNN OPENMP #OPENCV

DEFINES += $$OPTIONS



# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += \
    $$PWD/include \
    $$PWD/src

HEADERS += \
    include/darknet.h \
    src/activation_layer.h \
    src/activations.h \
    src/avgpool_layer.h \
    src/batchnorm_layer.h \
    src/blas.h \
    src/box.h \
    src/classifier.h \
    src/col2im.h \
    src/connected_layer.h \
    src/convolutional_layer.h \
    src/cost_layer.h \
    src/crnn_layer.h \
    src/crop_layer.h \
    src/cuda.h \
    src/data.h \
    src/deconvolutional_layer.h \
    src/demo.h \
    src/detection_layer.h \
    src/dropout_layer.h \
    src/gemm.h \
    src/gru_layer.h \
    src/im2col.h \
    src/image.h \
    src/layer.h \
    src/list.h \
    src/local_layer.h \
    src/lstm_layer.h \
    src/matrix.h \
    src/maxpool_layer.h \
    src/network.h \
    src/normalization_layer.h \
    src/option_list.h \
    src/parser.h \
    src/region_layer.h \
    src/reorg_layer.h \
    src/rnn_layer.h \
    src/route_layer.h \
    src/shortcut_layer.h \
    src/softmax_layer.h \
    src/stb_image.h \
    src/stb_image_write.h \
    src/tree.h \
    src/utils.h

LIBS += \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -lgomp \
    -lpthread

#######################################################################################################
# CUDA
CUDA_DIR = /usr/local/cuda
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bi

# libs used in your code
LIBS += -lcuda -lcudart -lcublas -lcurand -lcudnn

# GPU architecture
#CUDA_ARCH     = -arch=sm_30 \
#                -gencode=arch=compute_20,code=sm_20 \
#                -gencode=arch=compute_30,code=sm_30 \
#                -gencode=arch=compute_50,code=sm_50 \
#                -gencode=arch=compute_52,code=sm_52 \
#                -gencode=arch=compute_60,code=sm_60 \
#                -gencode=arch=compute_61,code=sm_61 \
#                -gencode=arch=compute_61,code=compute_61
CUDA_ARCH = -gencode arch=compute_61,code=[sm_61,compute_61]

# Here are some NVCC flags I've always used by default.
#NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
NVCCFLAGS     = --compiler-options

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

CUDA_OPT = $$join(OPTIONS,' -D','-D',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc $$CUDA_ARCH $$CUDA_OPT -m64 -O3 -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                #2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc $$ -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = $$OUT_PWD/${QMAKE_FILE_BASE}.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

# Cuda sources
CUDA_SOURCES = \
    src/activation_kernels.cu \
    src/avgpool_layer_kernels.cu \
    src/blas_kernels.cu \
    src/col2im_kernels.cu \
    src/convolutional_kernels.cu \
    src/crop_layer_kernels.cu \
    src/deconvolutional_kernels.cu \
    src/dropout_layer_kernels.cu \
    src/im2col_kernels.cu \
    src/maxpool_layer_kernels.cu
#######################################################################################################

SOURCES += \
    src/activation_layer.c \
    src/activations.c \
    src/avgpool_layer.c \
    src/batchnorm_layer.c \
    src/blas.c \
    src/box.c \
    src/col2im.c \
    #src/compare.c \
    src/connected_layer.c \
    src/convolutional_layer.c \
    src/cost_layer.c \
    src/crnn_layer.c \
    src/crop_layer.c \
    src/cuda.c \
    src/data.c \
    src/deconvolutional_layer.c \
    src/demo.c \
    src/detection_layer.c \
    src/dropout_layer.c \
    src/gemm.c \
    src/gru_layer.c \
    src/im2col.c \
    src/image.c \
    src/layer.c \
    src/list.c \
    src/local_layer.c \
    src/lstm_layer.c \
    src/matrix.c \
    src/maxpool_layer.c \
    src/network.c \
    src/normalization_layer.c \
    src/option_list.c \
    src/parser.c \
    src/region_layer.c \
    src/reorg_layer.c \
    src/rnn_layer.c \
    src/route_layer.c \
    src/shortcut_layer.c \
    src/softmax_layer.c \
    src/tree.c \
    src/utils.c \
    examples/art.c \
    examples/attention.c \
    examples/captcha.c \
    examples/cifar.c \
    examples/classifier.c \
    examples/coco.c \
    examples/darknet.c \
    examples/detector.c \
    #examples/dice.c \
    examples/go.c \
    examples/lsd.c \
    examples/nightmare.c \
    examples/regressor.c \
    examples/rnn.c \
    #examples/rnn_vid.c \
    examples/segmenter.c \
    examples/super.c \
    #examples/swag.c \
    examples/tag.c \
    #examples/voxel.c \
    #examples/writing.c \
    examples/yolo.c \
    $$CUDA_SOURCES

SOURCES -= $$CUDA_SOURCES # Remove from compilation, leaving in the project tree

QMAKE_CFLAGS    += -fopenmp -fPIC -Wno-unused-parameter
QMAKE_CXXFLAGS  += -fopenmp -fPIC -Wno-unused-parameter
QMAKE_LFLAGS    += -fopenmp





