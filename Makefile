ifeq ($(OS), Windows_NT)
	TARGET = a.exe
	OUTPUT = a.exe
else
	TARGET = a
	OUTPUT = a
endif


all: $(TARGET) out.csv

$(TARGET): main.cu
	nvcc -o $(OUTPUT) main.cu

out.csv: $(TARGET)
	./$(TARGET)