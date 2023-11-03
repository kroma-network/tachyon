#ifndef TACHYON_ZK_BASE_BLINDER_H_
#define TACHYON_ZK_BASE_BLINDER_H_

#include <stddef.h>

#include <vector>

#include "tachyon/zk/base/random_field_generator.h"

namespace tachyon::zk {

template <typename PCSTy>
class Blinder {
 public:
  using F = typename PCSTy::Field;
  using Evals = typename PCSTy::Evals;

  Blinder(RandomFieldGenerator<F>* random_field_generator,
          size_t blinding_factors)
      : random_field_generator_(random_field_generator),
        blinding_factors_(blinding_factors) {}

  const RandomFieldGenerator<F>* random_field_generator() const {
    return random_field_generator_;
  }
  size_t blinding_factors() const { return blinding_factors_; }

  // Blinds |evals| at behind by |blinding_factors_|.
  // Returns false if |evals.NumElements()| is less than |blinding_factors_|.
  bool Blind(Evals& evals) {
    size_t size = evals.NumElements();
    if (size < blinding_factors_) return false;
    size_t start = size - blinding_factors_;
    std::vector<F>& values = evals.evaluations();
    for (size_t i = start; i < size; ++i) {
      values[i] = random_field_generator_->Generate();
    }
    return true;
  }

 private:
  // not owned
  RandomFieldGenerator<F>* const random_field_generator_;
  size_t blinding_factors_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_BLINDER_H_
