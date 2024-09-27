#include "tachyon/c/base/profiler.h"

#include "tachyon/c/base/profiler_type_traits.h"

using namespace tachyon;

tachyon_profiler* tachyon_profiler_create(const char* path, size_t path_len) {
  return c::base::c_cast(
      new base::Profiler({base::FilePath(std::string_view(path, path_len))}));
}

void tachyon_profiler_destroy(tachyon_profiler* profiler) {
  delete c::base::native_cast(profiler);
}

void tachyon_profiler_init(tachyon_profiler* profiler) {
  c::base::native_cast(profiler)->Init();
}

void tachyon_profiler_start(tachyon_profiler* profiler) {
  c::base::native_cast(profiler)->Start();
}

void tachyon_profiler_stop(tachyon_profiler* profiler) {
  c::base::native_cast(profiler)->Stop();
}
