CFLAGS=-std=gnu++20 -Wall -Wextra -Wpedantic -Wno-unused-function -lm -march=native -shared -fPIC
CC=g++

.PHONY : clean release debug

dynamics: dynamics.cpp linalg.h
	$(CC) -o dynamics.so dynamics.cpp -Ofast  -s $(CFLAGS)

release : dynamics

debug : dynamics.cpp linalg.h
	$(CC) -o dynamics_debug.so dynamics.cpp -O0 -g $(CFLAGS)

clean :
	rm -f dynamics.so dynamics_debug.so
