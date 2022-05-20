include ../common-def.mk

TGT=libcommontest.a
SRC_C=*.cu
SRC_H=*.h
SRC=$(SRC_C) $(SRC_H)
TMP=$(TGT) *~ \#*

ARCH := -gencode arch=compute_61,code=sm_61

FLAGS= $(ARCH) -O3 -rdc=true -lib -I../include # -I../../../../Simulators/voxcraft/voxcraft-sim/3rdparty/thrust
ifeq ($(WITH_SCATTER), 1)
FLAGS += -DWITH_SCATTER
endif

build: $(TGT) 
$(TGT):	$(SRC) makefile
	nvcc $(FLAGS) -o $(TGT) $(SRC_C)

clean:
	rm -f $(TMP)
