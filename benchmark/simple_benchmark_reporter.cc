#include "benchmark/simple_benchmark_reporter.h"

#if defined(TACHYON_HAS_MATPLOTLIB)
#include "third_party/matplotlibcpp17/include/pyplot.h"

using namespace matplotlibcpp17;
#endif  // defined(TACHYON_HAS_MATPLOTLIB)

#include "absl/strings/str_cat.h"

#include "tachyon/base/console/table_writer.h"

namespace tachyon {

void SimpleBenchmarkReporter::Show() {
  base::TableWriterBuilder builder;
  base::TableWriter writer = builder.AlignHeaderLeft()
                                 .AddSpace(1)
                                 .FitToTerminalWidth()
                                 .StripTrailingAsciiWhitespace()
                                 .AddColumn("NAME")
                                 .AddColumn("TIME(sec)")
                                 .Build();
  for (size_t i = 0; i < results_.size(); ++i) {
    writer.SetElement(i, 0, names_[i]);
    writer.SetElement(i, 1, absl::StrCat(results_[i]));
  }
  writer.Print(true);

#if defined(TACHYON_HAS_MATPLOTLIB)
  py::scoped_interpreter guard{};
  auto plt = pyplot::import();

  auto [fig, ax] = plt.subplots(
      Kwargs("layout"_a = "constrained", "figsize"_a = py::make_tuple(12, 6)));

  ax.set_title(Args("Benchmark results"));

  ax.bar(Args(names_, results_));
  plt.show();
#endif  // defined(TACHYON_HAS_MATPLOTLIB)
}

}  // namespace tachyon
