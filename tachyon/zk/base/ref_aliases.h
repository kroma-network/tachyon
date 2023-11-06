#ifndef TACHYON_ZK_BASE_REF_ALIASES_H_
#define TACHYON_ZK_BASE_REF_ALIASES_H_

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/ref.h"

namespace tachyon::zk {

template <typename Point>
using PointRef = Ref<Point>;

template <typename Evals>
using EvalsRef = Ref<Evals>;

template <typename Poly>
using PolyRef = Ref<Poly>;

template <typename Poly>
using BlindedPolyRef = Ref<BlindedPolynomial<Poly>>;

template <typename Commitment>
using CommitmentRef = Ref<Commitment>;

template <typename Field>
using FieldRef = Ref<Field>;

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_REF_ALIASES_H_
