# Makefile for GPS+IMU Fusion SR-UKF Example

SRUKF_DIR = ../..
COMMON_DIR = ../common

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -O2 -I$(SRUKF_DIR) -I$(COMMON_DIR) -DHAVE_LAPACK -std=c99
LDFLAGS = -L$(SRUKF_DIR) -lsrukf -lm -llapacke -lblas -lopenblas -Wl,-rpath,$(SRUKF_DIR)

# Targets
TARGET = gps_imu
COMMON_OBJS = $(COMMON_DIR)/plot_utils.o

.PHONY: all clean run

all: $(TARGET)

# Build common utilities
$(COMMON_DIR)/plot_utils.o: $(COMMON_DIR)/plot_utils.c $(COMMON_DIR)/plot_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

# Build GPS+IMU example
$(TARGET): gps_imu.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(COMMON_OBJS) $(LDFLAGS)

# Run with default settings
run: $(TARGET)
	./$(TARGET) --format=svg --open

# Clean build artifacts and generated outputs
clean:
	rm -f $(TARGET) $(COMMON_OBJS)
	rm -f *.svg *.csv *.json

# Clean but restore pre-generated SVGs from git
clean-reset: clean
	git checkout -- *.svg 2>/dev/null || true
