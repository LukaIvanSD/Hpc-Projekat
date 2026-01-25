# Compiler
CC = gcc

# Target executable
TARGET = jato.exe

# Source files
SRCS = jato.c boid.c
OBJS = $(SRCS:.c=.o)

# SDL2 flags
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LIBS   = $(shell sdl2-config --libs)

# SDL2_ttf
TTF_LIBS = -lSDL2_ttf

# Compiler flags
CFLAGS = -Wall -Wextra -std=c11 -O2 $(SDL_CFLAGS)

# Linker flags
LDFLAGS = $(SDL_LIBS) $(TTF_LIBS) -lm

# Default rule
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
