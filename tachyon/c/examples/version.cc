#include "tachyon/c/version.h"

#include <stdio.h>

int main() {
  printf("Runtime Version: %u\n", tachyon_get_runtime_version());
  printf("Runtime Version String: %s\n", tachyon_get_runtime_version_str());
  printf("Runtime Full Version String: %s\n",
         tachyon_get_runtime_full_version_str());
  return 0;
}
