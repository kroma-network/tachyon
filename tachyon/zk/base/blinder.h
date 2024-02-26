#ifndef TACHYON_ZK_BASE_BLINDER_H_
#define TACHYON_ZK_BASE_BLINDER_H_

#include <stddef.h>

#include "tachyon/zk/base/random_field_generator_base.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk {

template <typename F>
class Blinder {
 public:
  Blinder(RandomFieldGeneratorBase<F>* random_field_generator,
          RowIndex blinding_factors)
      : random_field_generator_(random_field_generator),
        blinding_factors_(blinding_factors) {}

  const RandomFieldGeneratorBase<F>* random_field_generator() const {
    return random_field_generator_;
  }
  void set_blinding_factors(RowIndex blinding_factors) {
    blinding_factors_ = blinding_factors;
  }
  RowIndex blinding_factors() const { return blinding_factors_; }

  // The number of |blinding_rows| is determined to be either
  // |blinding_factors_| or |blinding_factors_| + 1, depending on the
  // |include_last_row| option.
  // Blinds |evals| at behind by |blinding_rows|.
  // Returns false if |evals.NumElements()| is less than |blinding_rows|.
  template <typename Evals>
  bool Blind(Evals& evals, bool include_last_row = false) {
    // NOTE(chokobole): It's safe to downcast because domain is already checked.
    RowIndex size = static_cast<RowIndex>(evals.NumElements());
    RowIndex blinding_rows = blinding_factors_;
    if (include_last_row) ++blinding_rows;
    if (size < blinding_rows) return false;
    RowIndex start = size - blinding_rows;
    for (RowIndex i = start; i < size; ++i) {
      // NOTE(chokobole): Boundary check is the responsibility of API callers.
      evals.at(i) = random_field_generator_->Generate();
    }
    return true;
  }

  F Generate() { return random_field_generator_->Generate(); }

 private:
  // not owned
  RandomFieldGeneratorBase<F>* random_field_generator_ = nullptr;
  RowIndex blinding_factors_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_BLINDER_H_
