WARNING_FLAGS=-Wall -Wextra -Wpedantic -Wno-unused-function
CFLAGS=-std=gnu++20 -shared -fPIC -march=native $(WARNING_FLAGS)
CLIBS=-lgsl -lgslcblas -lm
CXX=g++
SRC_DEPS=dynamics.h linalg.h solve.cpp events.h types.h

.PHONY : clean release debug solve

solve : $(SRC_DEPS)
	$(CXX) -o solve.so solve.cpp -Ofast -s $(CFLAGS) $(CLIBS)

release : dynamics

debug : $(SRC_DEPS)
	$(CXX) -o solve_debug.so solve.cpp -O0 -g $(CFLAGS) $(CLIBS)

clean :
	rm -f solve.so solve_debug.so
