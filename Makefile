CC = g++
OPT = -std=c++14
LIBS = -lopenblas -llapack -lfftw3 -lgmpxx -lgmp
TARGET = IPT

all : $(TARGET)

$(TARGET): IPT.o pade.o
	$(CC) -o $@ $^ $(LIBS) $(OPT)

IPT.o : IPT.cpp
	$(CC) -c -o $@ $^ $(OPT)

pade.o : pade.cpp
	$(CC) -c -o $@ $^

clean:
	rm -rf *.o $(TARGET)
