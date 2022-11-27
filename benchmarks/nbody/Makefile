include example.mk

CC=g++

LDIR =

OBJ = main.o

# -fopt-info-vec-missed
# -Rpass-analysis=loop-vectorize

%.o: %.cpp
	$(CC) -fopenmp -march=native -fopt-info-vec-missed  -mavx2 -ftree-vectorize -fno-math-errno -ffast-math  -O3 -g -c --std=c++17 -o $@ $< $(INCLUDE_PATH)

vect: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS_PATH) $(LIBS)

all: vect

run: all
	mpirun --oversubscribe -np 2 ./vect

debug:
	../../../gdbgui/launch_mpi_debugger 2 ./vect

.PHONY: clean all run

clean:
	rm -f *.o *~ core vect

