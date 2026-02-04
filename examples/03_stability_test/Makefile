# Makefile for Stability Test SR-UKF Example

SRUKF_DIR = ../..
COMMON_DIR = ../common

# Compiler and flags (use ?= to allow override for sanitizers)
CC ?= gcc
CFLAGS ?= -Wall -Wextra -Wpedantic -O2 -I$(SRUKF_DIR) -I$(COMMON_DIR) -DHAVE_LAPACK -std=c99
LDFLAGS ?= -L$(SRUKF_DIR) -lsrukf -lm -llapacke -lblas -lopenblas -Wl,-rpath,$(SRUKF_DIR)

# Targets
TARGET = stability
COMMON_OBJS = $(COMMON_DIR)/plot_utils.o

.PHONY: all clean run

all: $(TARGET)

# Build common utilities
$(COMMON_DIR)/plot_utils.o: $(COMMON_DIR)/plot_utils.c $(COMMON_DIR)/plot_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

# Build stability test
$(TARGET): stability.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(COMMON_OBJS) $(LDFLAGS)

# Run with default settings
run: $(TARGET)
	./$(TARGET) --scenario=baseline --verbose

# Clean build artifacts and generated outputs
clean:
	rm -f $(TARGET) $(COMMON_OBJS)
	rm -f *.svg *.csv *.json

# Clean but restore pre-generated SVGs from git
clean-reset: clean
	git checkout -- *.svg 2>/dev/null || true
