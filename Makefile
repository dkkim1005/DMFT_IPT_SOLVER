CC = g++
OPT = -std=c++14
LIBS = -lopenblas -llapack -lfftw3 -lgmpxx -lgmp -lboost_program_options
TARGET = IPT

all : $(TARGET)

$(TARGET): main.o pade.o
	$(CC) -o $@ $^ $(LIBS) $(OPT)

main.o : main.cpp
	$(CC) -c -o $@ $^ $(OPT)

pade.o : pade.cpp
	$(CC) -c -o $@ $^

clean:
	rm -rf *.o $(TARGET)
