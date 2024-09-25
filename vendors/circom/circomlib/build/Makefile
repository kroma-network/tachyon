CC = g++
CFLAGS = -std=c++11 -O3 -I. -I../../external/llvm-project/clang/staging/include
DEPS_HPP = circom.hpp calcwit.hpp fr.hpp functions.hpp

# Source Files
SRC = $(wildcard part_*.cpp)

# Object Files
OBJ = $(SRC:.cpp=.o)

# Additional Object Files
DEPS_O = main.o calcwit.o fr.o fr_asm.o

# NASM Compiler for Assembly File
ifeq ($(shell uname),Darwin)
    NASM = nasm -fmacho64 --prefix _
endif
ifeq ($(shell uname),Linux)
    NASM = nasm -felf64
endif

# Default Target
all: witness_generator

# Compilation Rules
%.o: %.cpp $(DEPS_HPP)
	$(CC) $(CFLAGS) -c $< -o $@

# Assembly Compilation
fr_asm.o: fr.asm
	$(NASM) fr.asm -o fr_asm.o

# Linking
witness_generator: $(DEPS_O) $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lgmp

# Clean
.PHONY: clean
clean:
	rm -f $(OBJ) $(DEPS_O) witness_generator
