# Define the source and object directories
SRC_DIR := neuralc
OBJ_DIR := build
OUT_DIR := bin

# Compiler and flags
CC := gcc
CFLAGS := -Wall -Wextra -I$(SRC_DIR)
LDFLAGS :=

# Find all .c files in the neuralc directory
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)

# Define the corresponding object files in the build directory
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Define the final executable name
TARGET := $(OUT_DIR)/neuralc

# Default target: build the executable
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJ_FILES)
	@mkdir -p $(OUT_DIR)
	$(CC) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)

# Compile the .c files to .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -rf $(OBJ_DIR) $(OUT_DIR)

.PHONY: all clean

