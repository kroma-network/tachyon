#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk/cuzk_ell_sparse_matrix.h"

#include <sstream>

#include "absl/strings/substitute.h"

#include "tachyon/base/console/table_writer.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon::math {

std::string CUZKELLSparseMatrix::ToString() const {
  std::stringstream ss;
  ss << "===ROW LENGTHS===" << std::endl;
  for (unsigned int i = 0; i < rows; ++i) {
    ss << row_lengths[i] << std::endl;
  }

  base::TableWriterBuilder builder;
  builder.AlignHeaderLeft()
      .AddSpace(1)
      .FitToTerminalWidth()
      .StripTrailingAsciiWhitespace();
  for (size_t i = 0; i < cols; ++i) {
    builder.AddColumn(absl::Substitute("COL($0)", i));
  }

  base::TableWriter writer = builder.Build();
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < row_lengths[i]; ++j) {
      writer.SetElement(i, j, base::NumberToString(col_indices[i * cols + j]));
    }
  }

  ss << "===COL INDICIES===" << std::endl;
  ss << writer.ToString() << std::endl;
  return ss.str();
}

}  // namespace tachyon::math
