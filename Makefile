SR_DIR      := $(CURDIR)

# Output directories
BIN_DIR     := $(SR_DIR)/bin

# Installation prefix
PREFIX      ?= /usr/local

# Flags
CFLAGS      := -Wall -Wextra -Wpedantic -O2 -fPIC -I$(SR_DIR) -DHAVE_LAPACK
LDFLAGS     := -lm -llapacke -lblas -lopenblas

LIB_SRCS    := srukf.c
LIB_HDRS    := srukf.h
LIB_NAME    := libsrukf.so

TEST_DIR    := $(CURDIR)/tests
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS   := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BIN_DIR)/%.out)
TEST_LD     := -L$(SR_DIR) -lsrukf -Wl,-rpath,$(SR_DIR) $(LDFLAGS)

# Tests that need internal access (include .c directly, don't link library)
INTERNAL_TESTS := 00_sigma 06_predict 10_simple 20_nonlinear 30_errors 35_numerical 40_stress 46_edge_cases

# Single-precision test (compiled with -DSRUKF_SINGLE)
SINGLE_PREC_TEST := 47_single_precision

# shared library target
$(LIB_NAME): $(LIB_SRCS) $(LIB_HDRS)
	$(CC) $(CFLAGS) -shared -Wl,-soname,$@ -o $@ $(LIB_SRCS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

# Internal tests: compile with srukf.c directly (no library link)
define INTERNAL_TEST_RULE
$(BIN_DIR)/$(1).out: $(TEST_DIR)/$(1).c $(LIB_SRCS) $(LIB_HDRS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $$@ $$< $(LDFLAGS)
endef
$(foreach t,$(INTERNAL_TESTS),$(eval $(call INTERNAL_TEST_RULE,$(t))))

# Single-precision test: compile with -DSRUKF_SINGLE
$(BIN_DIR)/$(SINGLE_PREC_TEST).out: $(TEST_DIR)/$(SINGLE_PREC_TEST).c $(LIB_SRCS) $(LIB_HDRS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DSRUKF_SINGLE -o $@ $< $(LDFLAGS)

# Public API tests: link against library
$(BIN_DIR)/%.out: $(TEST_DIR)/%.c $(LIB_NAME) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LD)

# Benchmarks
BENCH_DIR   := $(SR_DIR)/benchmark
BENCH_SRC   := $(BENCH_DIR)/benchmark.c
BENCH_BIN   := $(BIN_DIR)/benchmark.out
MEM_BENCH_SRC := $(BENCH_DIR)/memory_bench.c
MEM_BENCH_BIN := $(BIN_DIR)/memory_bench.out

$(BENCH_BIN): $(BENCH_SRC) $(LIB_NAME) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LD)

$(MEM_BENCH_BIN): $(MEM_BENCH_SRC) $(LIB_NAME) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LD)

.PHONY: all test test-verbose lib clean format bench bench-chart bench-memory bench-memory-chart install coverage docs
all: lib test

lib: $(LIB_NAME)

format:
	clang-format -i $(LIB_SRCS) $(LIB_HDRS) $(TEST_SRCS)

test: $(TEST_BINS)
	@failed=0; \
	for t in $(TEST_BINS); do \
	  name=$$(basename $$t .out); \
	  if $$t >/dev/null 2>&1; then \
	    echo "  $$name OK"; \
	  else \
	    echo "  $$name FAILED"; \
	    failed=1; \
	  fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
	  echo "All tests passed."; \
	else \
	  echo "Some tests failed."; \
	  exit 1; \
	fi

test-verbose: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
	  echo "=== $$t ==="; \
	  $$t || exit 1; \
	  echo; \
	done
	@echo "All tests passed."

bench: $(BENCH_BIN)
	$(BENCH_BIN)

bench-chart: $(BENCH_BIN)
	$(BENCH_BIN) | python3 benchmark/generate_chart.py > benchmark/benchmark.svg
	@echo "Generated benchmark/benchmark.svg"

bench-memory: $(MEM_BENCH_BIN)
	$(MEM_BENCH_BIN)

bench-memory-chart: $(MEM_BENCH_BIN)
	$(MEM_BENCH_BIN) | python3 benchmark/generate_memory_chart.py > benchmark/memory.svg
	@echo "Generated benchmark/memory.svg"

install: lib
	install -d $(PREFIX)/lib $(PREFIX)/include
	install -m 644 $(LIB_NAME) $(PREFIX)/lib/
	install -m 644 srukf.h $(PREFIX)/include/

coverage: CFLAGS += --coverage
coverage: LDFLAGS += --coverage
coverage: clean lib $(TEST_BINS)
	@for t in $(TEST_BINS); do $$t || true; done
	gcov srukf.c
	@echo "Coverage report in srukf.c.gcov"

docs:
	doxygen Doxyfile
	@echo "Documentation generated in docs/html/"

clean:
	rm -f $(LIB_NAME) $(TEST_BINS) $(BENCH_BIN) $(MEM_BENCH_BIN)
	rm -f *.gcno *.gcda *.gcov coverage.info
	rm -rf coverage-report docs
	rmdir --ignore-fail-on-non-empty $(BIN_DIR) 2>/dev/null || true
