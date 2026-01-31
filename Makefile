# =========================
# Compilers
# =========================
CC   = gcc
NVCC = nvcc

# =========================
# OpenMP flags
# =========================
OMP_CFLAGS  = -fopenmp
OMP_LDFLAGS = -Xcompiler -fopenmp

# =========================
# Target executable
# =========================
TARGET = jato.exe

# =========================
# Source files
# =========================
C_SRCS  = jato.c boid.c
CU_SRCS = boid_cuda.cu

# =========================
# Object files
# =========================
C_OBJS  = $(C_SRCS:.c=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)

OBJS = $(C_OBJS) $(CU_OBJS)

# =========================
# SDL2 flags
# =========================
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LIBS   = $(shell sdl2-config --libs)

# =========================
# SDL2_ttf
# =========================
TTF_LIBS = -lSDL2_ttf

# =========================
# Compiler flags
# =========================
CFLAGS  = -Wall -Wextra -std=c11 -O2 $(SDL_CFLAGS) $(OMP_CFLAGS)
NVFLAGS = -O2

# =========================
# Linker flags
# =========================
LDFLAGS = $(SDL_LIBS) $(TTF_LIBS) -lm -lcudart

# =========================
# Default rule
# =========================
all: $(TARGET)

# =========================
# Link (use nvcc!)
# =========================
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS) $(OMP_LDFLAGS)

# =========================
# Compile C files (gcc)
# =========================
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# =========================
# Compile CUDA files (nvcc)
# =========================
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# =========================
# Clean
# =========================
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
