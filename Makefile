SR_DIR      := $(CURDIR)
LAH_DIR     := $(SR_DIR)/../linear-algebra-helpers
LAH_INC     := $(LAH_DIR)/Include
LAH_LIB     := $(LAH_DIR)/Lib

# Output directories
BIN_DIR     := $(SR_DIR)/bin

# Flags
CFLAGS      := -Wall -Wextra -Wpedantic -O2 -fPIC -I$(LAH_INC) -I$(SR_DIR) -DHAVE_LAPACK
LDFLAGS     := -L$(LAH_LIB) -llah -Wl,-rpath,$(LAH_LIB) -lm -llapacke -lblas -lopenblas
LDLIBS      := -lsr_ukf -llah -lm -llapacke -lblas -lopenblas

LIB_SRCS    := sr_ukf.c
LIB_HDRS    := sr_ukf.h
LIB_NAME    := libsr_ukf.so

TEST_DIR    := $(CURDIR)/tests
TEST_SRCS   := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS   := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BIN_DIR)/%.out)
TEST_LD     := -L$(SR_DIR) -lsr_ukf -Wl,-rpath,$(SR_DIR) $(LDFLAGS)
TEST_OUTPUT := tests-output.txt

# Tests that need internal access (include .c directly, don't link library)
INTERNAL_TESTS := 00_sigma 03_gain 04_mean_cov 06_predict 10_simple 20_nonlinear 30_errors

# shared library target
$(LIB_NAME): $(LIB_SRCS) $(LIB_HDRS)
	$(CC) $(CFLAGS) -shared -Wl,-soname,$@ -o $@ $(LIB_SRCS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

# Internal tests: compile with sr_ukf.c directly (no library link)
define INTERNAL_TEST_RULE
$(BIN_DIR)/$(1).out: $(TEST_DIR)/$(1).c $(LIB_SRCS) $(LIB_HDRS) $(LAH_DIR)/Lib/liblah.so | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $$@ $$< $(LDFLAGS)
endef
$(foreach t,$(INTERNAL_TESTS),$(eval $(call INTERNAL_TEST_RULE,$(t))))

# Public API tests: link against library
$(BIN_DIR)/%.out: $(TEST_DIR)/%.c $(LIB_NAME) $(LAH_DIR)/Lib/liblah.so | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(TEST_LD)

.PHONY: all test lib clean format tests-output.txt
all: lib test

lib: $(LIB_NAME)

format:
	clang-format -i $(LIB_SRCS) $(LIB_HDRS) $(TEST_SRCS)

$(TEST_OUTPUT): $(LIB_NAME) $(TEST_BINS)
	@echo "Running unit tests..."
	@rm -f $@
	@for t in $(TEST_BINS); do \
	  echo "=== $$t ===" >> $@; \
	  $${t} >> $@ 2>&1; \
		echo "return code: $$?" >> $@; \
	done
	@echo "Output written to $@."
	@cat $@

test: $(TEST_BINS)
	make tests-output.txt
	cat tests-output.txt
	@echo "Running unit tests..."
	@for t in $(TEST_BINS); do \
	  echo "=== $$t ==="; \
	  $${t} || exit 1; \
	  echo; \
	done
	@echo "All tests passed."

clean:
	rm -f $(LIB_NAME) $(TEST_BINS)
	rmdir --ignore-fail-on-non-empty $(BIN_DIR)
