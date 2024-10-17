CFLAGS=-std=gnu++20 -Wall -Wextra -Wpedantic -Wno-unused-function -shared -fPIC -march=native
CLIBS=-lgsl -lgslcblas -lm
CC=g++

.PHONY : clean release debug solve


solve : $(DYNAMICS_SRC)
	$(CC) -o solve.so solve.cpp -Ofast  -s $(CFLAGS) $(CLIBS)

release : dynamics

debug : $(DYNAMICS_SRC)
	$(CC) -o dynamics_debug.so dynamics.cpp -O0 -g $(CFLAGS) $(CLIBS)

clean :
	rm -f dynamics.so dynamics_debug.so
