COMFLAGS = -lm -Wno-incompatible-pointer-types

all: a out.csv

a: main.cu
	nvcc -o a main.cu
#	clang $(COMFLAGS) -o a main.c

out.csv: a
	./a