#ifndef TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELL_H_
#define TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELL_H_

#include <string>

#include "tachyon/export.h"
#include "tachyon/zk/base/rotation.h"
#include "tachyon/zk/plonk/base/column_key.h"

namespace tachyon::zk::plonk {

class TACHYON_EXPORT VirtualCell {
 public:
  VirtualCell() = default;
  VirtualCell(const AnyColumnKey& column, Rotation rotation)
      : column_(column), rotation_(rotation) {}

  const AnyColumnKey& column() const { return column_; }
  Rotation rotation() const { return rotation_; }

  bool operator==(const VirtualCell& other) const {
    return column_ == other.column_ && rotation_ == other.rotation_;
  }
  bool operator!=(const VirtualCell& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("column: $0, rotation: $1", column_.ToString(),
                            rotation_.ToString());
  }

 private:
  AnyColumnKey column_;
  Rotation rotation_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_CONSTRAINT_SYSTEM_VIRTUAL_CELL_H_
