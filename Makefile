COMFLAGS = -lm

# ifeq ($(OS), Windows_NT)
# 	LINFLAGS = -L./lib -lvulkan-1 -lglfw3_mt -lgdi32 -luser32 -lshell32
# 	OUTPUT = VulkanTest.exe
# else
# 	LINFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
# 	OUTPUT = VulkanTest
# endif

PREFLAGS = -I./include
INPUTS = main.cpp


all: a out.csv

a: main.c
	clang $(COMFLAGS) -o a main.c

out.csv: a
	./a