// --------------------------------------------------------------------
// 90_cpp_linkage.cpp - Verifies the public header is consumable from
// C++: the extern "C" guards must prevent name mangling so a C++
// translation unit links against the C library.
//
// This test links against the public API. It is intentionally minimal:
// its value is that it compiles and links at all.
// --------------------------------------------------------------------

#include <cassert>
#include <cstring>

#include "srukf.h"

int main() {
  assert(std::strcmp(srukf_version(), SRUKF_VERSION) == 0);

  srukf *ukf = srukf_create(2, 1);
  assert(ukf != nullptr);
  assert(srukf_state_dim(ukf) == 2);
  assert(srukf_meas_dim(ukf) == 1);
  srukf_free(ukf);

  return 0;
}
