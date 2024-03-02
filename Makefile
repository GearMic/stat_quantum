ifeq ($(OS), Windows_NT)
	TARGET = a.exe
	OUTPUT = a.exe
else
	TARGET = a
	OUTPUT = a
endif

OBJECTS = main.o metropolis.o helper.o

all: $(OBJECTS) $(TARGET) out.csv 

%.o: %.cu 
	nvcc -dc -I. -o $@ $<

$(TARGET): main.cu
	nvcc -I. -o $(OUTPUT) $(OBJECTS)

out.csv: $(TARGET)
	./$(TARGET)