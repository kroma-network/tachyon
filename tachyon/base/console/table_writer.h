#ifndef TACHYON_BASE_CONSOLE_TABLE_WRITER_H_
#define TACHYON_BASE_CONSOLE_TABLE_WRITER_H_

#include <stddef.h>

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/export.h"

namespace tachyon {
namespace base {

class TACHYON_EXPORT TableWriter {
 public:
  enum class Align {
    kLeft,
    kRight,
    kCenter,
  };

  enum class Length {
    kAuto,
    kFixed,
  };

  enum class StripMode {
    kBothAsciiWhitespace,
    kTrailingAsciiWhitespace,
    kLeadingAsciiWhitespace,
  };

  TableWriter(const TableWriter& other);
  TableWriter& operator=(const TableWriter& other);
  TableWriter(TableWriter&& other) noexcept;
  TableWriter& operator=(TableWriter&& other) noexcept;
  ~TableWriter();

  struct ColumnWidth {
    Length length;
    size_t width;
  };

  void SetElement(size_t row, size_t col, std::string_view element);
  std::string ToString() const;

  void Print(bool with_new_line = true) const;

 private:
  friend class TableWriterBuilder;
  TableWriter();

  void AppendTable(std::ostream& os) const;
  void AppendRow(std::ostream& ss, absl::Span<const std::string> contents,
                 const std::vector<ColumnWidth>* column_widths,
                 Align align) const;

  Align header_align_ = Align::kLeft;
  Align body_align_ = Align::kLeft;
  size_t space_ = 0;
  StripMode strip_mode_ = StripMode::kBothAsciiWhitespace;
  bool fit_to_terminal_width_ = false;
  std::vector<std::string> titles_;
  std::vector<std::vector<std::string>> elements_;
  std::vector<ColumnWidth> column_widths_;
};

class TACHYON_EXPORT TableWriterBuilder {
 public:
  TableWriterBuilder();
  TableWriterBuilder(const TableWriterBuilder& other) = delete;
  TableWriterBuilder& operator=(const TableWriterBuilder& other) = delete;
  ~TableWriterBuilder();

  TableWriterBuilder& AlignHeaderLeft();
  TableWriterBuilder& AlignHeaderRight();
  TableWriterBuilder& AlignHeaderCenter();
  TableWriterBuilder& AlignBodyLeft();
  TableWriterBuilder& AlignBodyRight();
  TableWriterBuilder& AlignBodyCenter();
  TableWriterBuilder& AddSpace(size_t space);
  TableWriterBuilder& FitToTerminalWidth();
  TableWriterBuilder& StripBothAsciiWhitespace();
  TableWriterBuilder& StripTrailingAsciiWhitespace();
  TableWriterBuilder& StripLeadingAsciiWhitespace();
  TableWriterBuilder& AddColumn(std::string_view title);
  TableWriterBuilder& AddColumn(std::string_view title, size_t width);

  TableWriter Build() const;

 private:
  TableWriter writer_;
};

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os,
                                        const TableWriter& table_writer);

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_CONSOLE_TABLE_WRITER_H_
