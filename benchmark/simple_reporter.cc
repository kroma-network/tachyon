#include "benchmark/simple_reporter.h"

#if defined(TACHYON_HAS_MATPLOTLIB)
#include "third_party/matplotlibcpp17/include/pyplot.h"

using namespace matplotlibcpp17;
#endif  // defined(TACHYON_HAS_MATPLOTLIB)

// clang-format off
#include "benchmark/vendor.h"
// clang-format on
#include "tachyon/base/console/table_writer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/time/time.h"

namespace tachyon::benchmark {

void SimpleReporter::AddTime(Vendor vendor, base::TimeDelta time_taken) {
  measurements_[vendor].push_back(time_taken);
}

void SimpleReporter::AddVendor(Vendor vendor) { vendors_.push_back(vendor); }

void SimpleReporter::AddAverageAsLastColumn() {
  for (Vendor vendor : vendors_) {
    base::TimeDelta total =
        std::accumulate(measurements_[vendor].begin(),
                        measurements_[vendor].end(), base::TimeDelta());
    AddTime(vendor, total / measurements_[vendor].size());
  }
  column_labels_.push_back("avg");
}

void SimpleReporter::Show() {
  base::TableWriterBuilder builder;
  builder.AlignHeaderLeft()
      .AddSpace(1)
      .FitToTerminalWidth()
      .StripTrailingAsciiWhitespace()
      .AddColumn("");
  for (Vendor vendor : vendors_) {
    builder.AddColumn(vendor.ToString());
  }
  base::TableWriter writer = builder.Build();

  for (size_t i = 0; i < column_labels_.size(); ++i) {
    writer.SetElement(i, 0, column_labels_[i]);
    for (size_t j = 0; j < vendors_.size(); ++j) {
      writer.SetElement(
          i, j + 1,
          base::NumberToString(measurements_[vendors_[j]][i].InSecondsF()));
    }
  }
  writer.Print(true);

#if defined(TACHYON_HAS_MATPLOTLIB)
  py::scoped_interpreter guard{};
  auto plt = pyplot::import();

  const double kBarWidth = 1.0 / (vendors_.size() + 1);

  std::vector<size_t> x_positions =
      base::CreateRangedVector(size_t{0}, column_labels_.size());

  auto [fig, ax] = plt.subplots(Kwargs("layout"_a = "constrained"));

  for (size_t i = 0; i < vendors_.size(); ++i) {
    double offset = kBarWidth * i;
    std::vector<double> values =
        base::Map(measurements_[vendors_[i]],
                  [](base::TimeDelta delta) { return delta.InSecondsF(); });
    auto rects = ax.bar(
        Args(py::reinterpret_borrow<py::tuple>(py::cast(base::Map(
                 x_positions,
                 [offset](size_t x_position) { return x_position + offset; }))),
             py::reinterpret_borrow<py::tuple>(py::cast(values)), kBarWidth),
        Kwargs("label"_a = vendors_[i].ToString()));
  }

  ax.set_title(Args(title_));
  ax.set_xticks(
      Args(py::reinterpret_borrow<py::tuple>(py::cast(x_positions)),
           py::reinterpret_borrow<py::tuple>(py::cast(column_labels_))));
  ax.set_xlabel(Args(x_label_));
  ax.set_ylabel(Args(y_label_));
  ax.legend();

  plt.show();
#endif  // defined(TACHYON_HAS_MATPLOTLIB)
}

}  // namespace tachyon::benchmark
