#ifndef TACHYON_BASE_PROFILER_H_
#define TACHYON_BASE_PROFILER_H_

#include <memory>
#include <string>

#include "third_party/perfetto/perfetto.h"

#include "tachyon/base/files/file.h"
#include "tachyon/export.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("Utils").SetDescription("Base utility functions"),
    perfetto::Category("Subtask").SetDescription(
        "Subtask within a bigger task"),
    perfetto::Category("MSM").SetDescription(
        "Multi Scalar Multiplication operations"),
    perfetto::Category("ProofGeneration")
        .SetDescription("The proof generation process"),
    perfetto::Category("ProofVerification")
        .SetDescription("The proof verification process"),
    perfetto::Category("EvaluationDomain")
        .SetDescription("Evaluation Domain operations"));

namespace tachyon::base {

class TACHYON_EXPORT Profiler {
 public:
  struct Options {
    constexpr static size_t kDefaultMaxSizeKB = 1e6;

    base::FilePath output_path = base::FilePath("/tmp/tachyon.perfetto-trace");
    size_t max_size_kb = kDefaultMaxSizeKB;
  };

  Profiler();
  explicit Profiler(const Options& options);
  ~Profiler();

  void Init();
  void DisableCategories(std::string_view category);
  void EnableCategories(std::string_view category);
  void Start();
  void Stop();

 private:
  perfetto::protos::gen::TrackEventConfig track_event_cfg_;
  std::unique_ptr<perfetto::TracingSession> tracing_session_;
  FilePath trace_filepath_;
  File trace_file_;
  size_t max_size_kb_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_PROFILER_H_
