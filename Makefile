SR_DIR      := $(CURDIR)

# Output directories
BIN_DIR     := $(SR_DIR)/bin

# Installation prefix (DESTDIR is honored for staged installs)
PREFIX      ?= /usr/local
LIBDIR      ?= $(PREFIX)/lib
INCLUDEDIR  ?= $(PREFIX)/include
PCDIR       ?= $(LIBDIR)/pkgconfig

# The version is owned by srukf.h (see CONTRIBUTING.md); parse it.
# The leading dot in the pattern dodges make's comment character.
SRUKF_VERSION := $(shell sed -n 's/^.define SRUKF_VERSION "\(.*\)"/\1/p' srukf.h)
SRUKF_MAJOR   := $(word 1,$(subst ., ,$(SRUKF_VERSION)))

# Dependencies resolve through pkg-config, mirroring CMake's
# cblas -> blas -> openblas fallback chain. The static fallback is the
# historical link line for systems without pkg-config. Both CFLAGS and
# LDFLAGS use ?= so command-line/environment overrides win untouched.
BLAS_PC := $(shell for p in cblas blas openblas; do \
	if pkg-config --exists lapacke $$p 2>/dev/null; then echo $$p; break; fi; \
	done)
ifneq ($(BLAS_PC),)
PKG_CFLAGS := $(shell pkg-config --cflags lapacke $(BLAS_PC))
PKG_LIBS   := $(shell pkg-config --libs lapacke $(BLAS_PC))
else
PKG_CFLAGS :=
PKG_LIBS   := -llapacke -lblas -lopenblas
endif

CFLAGS      ?= -Wall -Wextra -Wpedantic -O2 -fPIC -I$(SR_DIR) -DHAVE_LAPACK $(PKG_CFLAGS)
LDFLAGS     ?= $(PKG_LIBS) -lm

LIB_SRCS    := srukf.c
LIB_PARTS   := $(wildcard src/*.c)
LIB_HDRS    := srukf.h
LIB_NAME    := libsrukf.so
LIB_A       := libsrukf.a

TEST_DIR    := $(CURDIR)/tests
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS   := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BIN_DIR)/%.out)
TEST_LD     := -L$(SR_DIR) -lsrukf -Wl,-rpath,$(SR_DIR) $(LDFLAGS)

# Tests are classified by convention (see CONTRIBUTING.md): a test that
# does `#include "srukf.c"` compiles the library source directly and
# must not also link it; everything else links the library. Adding a
# test requires no edits here.
INTERNAL_TESTS := $(basename $(notdir $(shell grep -sl 'include "srukf.c"' $(TEST_DIR)/*.c)))

# shared library target (soname carries the major version, like CMake's)
# srukf.c is a single translation unit that #includes src/*.c.
$(LIB_NAME): $(LIB_SRCS) $(LIB_PARTS) $(LIB_HDRS)
	$(CC) $(CFLAGS) -shared -Wl,-soname,$(LIB_NAME).$(SRUKF_MAJOR) -o $@ $(LIB_SRCS) $(LDFLAGS)
	ln -sf $(LIB_NAME) $(LIB_NAME).$(SRUKF_MAJOR)

# static library target
$(LIB_A): $(LIB_SRCS) $(LIB_PARTS) $(LIB_HDRS)
	$(CC) $(CFLAGS) -c $(LIB_SRCS) -o srukf.o
	$(AR) rcs $@ srukf.o

$(BIN_DIR):
	mkdir -p $@

# Internal tests: compile with srukf.c directly (no library link)
define INTERNAL_TEST_RULE
$(BIN_DIR)/$(1).out: $(TEST_DIR)/$(1).c $(LIB_SRCS) $(LIB_PARTS) $(LIB_HDRS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $$@ $$< $(LDFLAGS)
endef
$(foreach t,$(INTERNAL_TESTS),$(eval $(call INTERNAL_TEST_RULE,$(t))))

# Public API tests: link against library
$(BIN_DIR)/%.out: $(TEST_DIR)/%.c $(LIB_NAME) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LD)

# C++ linkage test: verifies the extern "C" guards in the public header.
# Only built when a C++ compiler is available (stripped-down systems may
# not carry one).
CXX ?= g++
ifneq ($(shell command -v $(CXX) 2>/dev/null),)
TEST_BINS += $(BIN_DIR)/90_cpp_linkage.out
$(BIN_DIR)/90_cpp_linkage.out: $(TEST_DIR)/90_cpp_linkage.cpp $(LIB_NAME) | $(BIN_DIR)
	$(CXX) $(CFLAGS) -o $@ $< $(TEST_LD)
endif

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

.PHONY: all test test-verbose count-tests lib clean format bench bench-chart bench-memory bench-memory-chart install coverage docs docs-serve docs-clean
all: lib test

lib: $(LIB_NAME) $(LIB_A)

format:
	clang-format -i $(LIB_SRCS) $(LIB_PARTS) $(LIB_HDRS) $(TEST_SRCS)

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

# Number of test binaries this Makefile would build; CI compares this
# against CTest's count to catch make/CMake drift.
count-tests:
	@echo $(words $(TEST_BINS))

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

# pkg-config file: same template CMake uses, filled by sed
srukf.pc: srukf.pc.in srukf.h
	sed -e 's|@CMAKE_INSTALL_PREFIX@|$(PREFIX)|' \
	    -e 's|@CMAKE_INSTALL_LIBDIR@|lib|' \
	    -e 's|@CMAKE_INSTALL_INCLUDEDIR@|include|' \
	    -e 's|@PROJECT_DESCRIPTION@|Square-Root Unscented Kalman Filter|' \
	    -e 's|@PROJECT_VERSION@|$(SRUKF_VERSION)|' \
	    $< > $@

install: lib srukf.pc
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PCDIR)
	install -m 755 $(LIB_NAME) $(DESTDIR)$(LIBDIR)/$(LIB_NAME).$(SRUKF_VERSION)
	ln -sf $(LIB_NAME).$(SRUKF_VERSION) $(DESTDIR)$(LIBDIR)/$(LIB_NAME).$(SRUKF_MAJOR)
	ln -sf $(LIB_NAME).$(SRUKF_MAJOR) $(DESTDIR)$(LIBDIR)/$(LIB_NAME)
	install -m 644 $(LIB_A) $(DESTDIR)$(LIBDIR)/
	install -m 644 srukf.h $(DESTDIR)$(INCLUDEDIR)/
	install -m 644 srukf.pc $(DESTDIR)$(PCDIR)/

coverage: CFLAGS += --coverage
coverage: LDFLAGS += --coverage
coverage: clean lib $(TEST_BINS)
	@for t in $(TEST_BINS); do $$t || true; done
	@# gcov produces .gcov files from .gcda runtime data
	@# Since srukf.c is compiled into each test, we'll have multiple .gcda files
	gcov srukf.c 2>/dev/null || true
	@echo "Coverage data collected. Use 'lcov --capture --directory . --output-file coverage.info' for detailed reports."

docs:
	@echo "Generating Doxygen documentation (HTML + Markdown)..."
	doxygen docs/Doxyfile
	@echo "Building MkDocs site..."
	mkdocs build -f docs/mkdocs.yml
	@echo "✓ Documentation generated:"
	@echo "  Landing page: docs/index.html"
	@echo "  Doxygen HTML: docs/doxygen/html/"
	@echo "  MkDocs site:  docs/mkdocs/site/"

docs-serve:
	@echo "Starting local documentation preview..."
	@echo "  Landing page: http://localhost:8000/"
	@echo "  Doxygen:      http://localhost:8000/doxygen/html/"
	@echo "  MkDocs:       http://localhost:8001/"
	@echo ""
	@(cd docs && python3 -m http.server 8000 > /dev/null 2>&1) & \
	mkdocs serve -f docs/mkdocs.yml -a localhost:8001

docs-clean:
	rm -rf docs/doxygen docs/mkdocs

clean:
	rm -f $(LIB_NAME) $(LIB_NAME).$(SRUKF_MAJOR) $(LIB_A) srukf.o srukf.pc
	rm -f $(TEST_BINS) $(BENCH_BIN) $(MEM_BENCH_BIN)
	rm -f *.gcno *.gcda *.gcov coverage.info
	rm -rf coverage-report docs/doxygen docs/mkdocs
	rmdir --ignore-fail-on-non-empty $(BIN_DIR) 2>/dev/null || true
