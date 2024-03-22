#include "benchmark/simple_benchmark_reporter.h"

#if defined(TACHYON_HAS_MATPLOTLIB)
#include "third_party/matplotlibcpp17/include/pyplot.h"

using namespace matplotlibcpp17;
#endif  // defined(TACHYON_HAS_MATPLOTLIB)

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"

#include "tachyon/base/console/table_writer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon {

void SimpleBenchmarkReporter::Show() {
  base::TableWriterBuilder builder;
  builder.AlignHeaderLeft()
      .AddSpace(1)
      .FitToTerminalWidth()
      .StripTrailingAsciiWhitespace()
      .AddColumn("");
  for (size_t i = 0; i < column_headers_.size(); ++i) {
    builder.AddColumn(column_headers_[i]);
  }
  base::TableWriter writer = builder.Build();

  for (size_t i = 0; i < targets_.size(); ++i) {
    writer.SetElement(i, 0, targets_[i]);
    for (size_t j = 0; j < column_headers_.size(); ++j) {
      writer.SetElement(i, j + 1, base::NumberToString(times_[i][j]));
    }
  }
  writer.Print(true);

#if defined(TACHYON_HAS_MATPLOTLIB)
  py::scoped_interpreter guard{};
  auto plt = pyplot::import();

  const double kBarWidth = 1.0 / (times_[0].size() + 1);

  std::vector<size_t> x_positions =
      base::CreateRangedVector(static_cast<size_t>(0), targets_.size());

  auto [fig, ax] = plt.subplots(Kwargs("layout"_a = "constrained"));

  for (size_t i = 0; i < column_headers_.size(); ++i) {
    double offset = kBarWidth * i;
    std::vector<double> values;
    for (size_t j = 0; j < targets_.size(); ++j) {
      values.push_back(times_[j][i]);
    }
    auto rects = ax.bar(
        Args(py::reinterpret_borrow<py::tuple>(py::cast(base::Map(
                 x_positions,
                 [offset](size_t x_position) { return x_position + offset; }))),
             py::reinterpret_borrow<py::tuple>(py::cast(values)), kBarWidth),
        Kwargs("label"_a = column_headers_[i]));
  }

  ax.set_title(Args(title_));
  ax.set_xticks(Args(py::reinterpret_borrow<py::tuple>(py::cast(x_positions)),
                     py::reinterpret_borrow<py::tuple>(py::cast(targets_))));
  ax.set_ylabel(Args("Time (sec)"));
  ax.legend();

  plt.show();
#endif  // defined(TACHYON_HAS_MATPLOTLIB)
}

}  // namespace tachyon
