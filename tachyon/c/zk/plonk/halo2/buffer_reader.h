#ifndef TACHYON_C_ZK_PLONK_HALO2_BUFFER_READER_H_
#define TACHYON_C_ZK_PLONK_HALO2_BUFFER_READER_H_

#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"

#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/zk/plonk/halo2/bn254_gwc_pcs.h"
#include "tachyon/c/zk/plonk/halo2/bn254_shplonk_pcs.h"
#include "tachyon/math/finite_fields/cubic_extension_field.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/finite_fields/quadratic_extension_field.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/lookup/argument.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/base/phase.h"
#include "tachyon/zk/plonk/constraint_system/challenge.h"
#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/constraint_system/lookup_tracker.h"
#include "tachyon/zk/plonk/constraint_system/query.h"
#include "tachyon/zk/plonk/expressions/expression_factory.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/shuffle/argument.h"

namespace tachyon::c::zk::plonk {

template <typename T, typename SFINAE = void>
class BufferReader;

template <typename T>
void ReadBuffer(const tachyon::base::ReadOnlyBuffer& buffer, T& value) {
  value = BufferReader<T>::Read(buffer);
}

template <typename T>
class BufferReader<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  static T Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    tachyon::base::EndianAutoReset resetter(buffer,
                                            tachyon::base::Endian::kBig);
    T v;
    CHECK(buffer.Read(&v));
    return v;
  }
};

inline bool ReadU8AsBool(const tachyon::base::ReadOnlyBuffer& buffer) {
  return BufferReader<uint8_t>::Read(buffer) != 0;
}

inline size_t ReadU32AsSizeT(const tachyon::base::ReadOnlyBuffer& buffer) {
  return size_t{BufferReader<uint32_t>::Read(buffer)};
}

template <>
class BufferReader<std::string> {
 public:
  static std::string Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    size_t size = ReadU32AsSizeT(buffer);
    std::string ret;
    ret.resize(size);
    CHECK(buffer.Read(reinterpret_cast<uint8_t*>(const_cast<char*>(ret.data())),
                      size));
    return ret;
  }
};

template <typename T>
class BufferReader<std::optional<T>> {
 public:
  static std::optional<T> Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    bool has_value = ReadU8AsBool(buffer);
    if (has_value) {
      return BufferReader<T>::Read(buffer);
    } else {
      return std::nullopt;
    }
  }
};

template <>
class BufferReader<std::optional<size_t>> {
 public:
  static std::optional<size_t> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    bool has_value = ReadU8AsBool(buffer);
    if (has_value) {
      return ReadU32AsSizeT(buffer);
    } else {
      return std::nullopt;
    }
  }
};

template <typename T>
class BufferReader<std::vector<T>> {
 public:
  static std::vector<T> Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    return tachyon::base::CreateVector(ReadU32AsSizeT(buffer), [&buffer]() {
      return BufferReader<T>::Read(buffer);
    });
  }
};

template <typename T>
class BufferReader<std::pmr::vector<T>> {
 public:
  static std::pmr::vector<T> Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    return tachyon::base::CreatePmrVector(ReadU32AsSizeT(buffer), [&buffer]() {
      return BufferReader<T>::Read(buffer);
    });
  }
};

template <typename K, typename V>
class BufferReader<absl::btree_map<K, V>> {
 public:
  static absl::btree_map<K, V> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    size_t size = ReadU32AsSizeT(buffer);
    absl::btree_map<K, V> ret;
    for (size_t i = 0; i < size; ++i) {
      K key = BufferReader<K>::Read(buffer);
      V value = BufferReader<V>::Read(buffer);
      ret[std::move(key)] = std::move(value);
    }
    return ret;
  }
};

template <typename Curve>
class BufferReader<tachyon::math::AffinePoint<Curve>> {
 public:
  using BaseField = typename tachyon::math::AffinePoint<Curve>::BaseField;

  static tachyon::math::AffinePoint<Curve> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    return {BufferReader<BaseField>::Read(buffer),
            BufferReader<BaseField>::Read(buffer)};
  }
};

template <typename T>
class BufferReader<
    T,
    std::enable_if_t<std::is_base_of_v<tachyon::math::PrimeFieldBase<T>, T>>> {
 public:
  using BigInt = typename T::BigIntTy;

  static T Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    tachyon::base::EndianAutoReset resetter(buffer,
                                            tachyon::base::Endian::kLittle);
    BigInt montgomery;
    CHECK(buffer.Read(montgomery.limbs));
    return T::FromMontgomery(montgomery);
  }
};

template <typename T>
class BufferReader<T, std::enable_if_t<std::is_base_of_v<
                          tachyon::math::QuadraticExtensionField<T>, T>>> {
 public:
  using BaseField = typename T::BaseField;

  static T Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    tachyon::base::EndianAutoReset resetter(buffer,
                                            tachyon::base::Endian::kLittle);
    BaseField c0 = BufferReader<BaseField>::Read(buffer);
    BaseField c1 = BufferReader<BaseField>::Read(buffer);
    return {std::move(c0), std::move(c1)};
  }
};

template <typename T>
class BufferReader<T, std::enable_if_t<std::is_base_of_v<
                          tachyon::math::CubicExtensionField<T>, T>>> {
 public:
  using BaseField = typename T::BaseField;

  static T Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    tachyon::base::EndianAutoReset resetter(buffer,
                                            tachyon::base::Endian::kLittle);
    BaseField c0 = BufferReader<BaseField>::Read(buffer);
    BaseField c1 = BufferReader<BaseField>::Read(buffer);
    BaseField c2 = BufferReader<BaseField>::Read(buffer);
    return {std::move(c0), std::move(c1), std::move(c1)};
  }
};

template <typename F, size_t MaxDegree>
class BufferReader<tachyon::math::UnivariateDensePolynomial<F, MaxDegree>> {
 public:
  static tachyon::math::UnivariateDensePolynomial<F, MaxDegree> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::pmr::vector<F> coeffs;
    ReadBuffer(buffer, coeffs);
    return tachyon::math::UnivariateDensePolynomial<F, MaxDegree>(
        tachyon::math::UnivariateDenseCoefficients<F, MaxDegree>(
            std::move(coeffs), true));
  }
};  // namespace tachyon::c::zk

template <typename F, size_t MaxDegree>
class BufferReader<tachyon::math::UnivariateEvaluations<F, MaxDegree>> {
 public:
  static tachyon::math::UnivariateEvaluations<F, MaxDegree> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::pmr::vector<F> evals;
    ReadBuffer(buffer, evals);
    return tachyon::math::UnivariateEvaluations<F, MaxDegree>(std::move(evals));
  }
};  // namespace tachyon::c::zk

template <>
class BufferReader<tachyon::zk::plonk::Phase> {
 public:
  static tachyon::zk::plonk::Phase Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    return tachyon::zk::plonk::Phase(BufferReader<uint8_t>::Read(buffer));
  }
};

template <>
class BufferReader<tachyon::zk::plonk::Challenge> {
 public:
  static tachyon::zk::plonk::Challenge Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    tachyon::zk::plonk::Phase phase =
        BufferReader<tachyon::zk::plonk::Phase>::Read(buffer);
    return {index, phase};
  }
};

template <>
class BufferReader<tachyon::zk::Rotation> {
 public:
  static tachyon::zk::Rotation Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    return tachyon::zk::Rotation(BufferReader<int32_t>::Read(buffer));
  }
};

template <>
class BufferReader<tachyon::zk::plonk::Selector> {
 public:
  static tachyon::zk::plonk::Selector Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    bool is_simple = ReadU8AsBool(buffer);
    return is_simple ? tachyon::zk::plonk::Selector::Simple(index)
                     : tachyon::zk::plonk::Selector::Complex(index);
  }
};

template <tachyon::zk::plonk::ColumnType C>
class BufferReader<tachyon::zk::plonk::ColumnKey<C>> {
 public:
  static tachyon::zk::plonk::ColumnKey<C> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    uint8_t kind = BufferReader<uint8_t>::Read(buffer);
    if constexpr (C == tachyon::zk::plonk::ColumnType::kAdvice) {
      CHECK_EQ(kind, static_cast<int8_t>(C));
      return tachyon::zk::plonk::ColumnKey<C>(
          index, BufferReader<tachyon::zk::plonk::Phase>::Read(buffer));
    } else {
      if constexpr (C == tachyon::zk::plonk::ColumnType::kInstance ||
                    C == tachyon::zk::plonk::ColumnType::kFixed) {
        CHECK_EQ(kind, static_cast<int8_t>(C));
        return tachyon::zk::plonk::ColumnKey<C>(index);
      } else {
        tachyon::zk::plonk::Phase phase =
            BufferReader<tachyon::zk::plonk::Phase>::Read(buffer);
        switch (static_cast<tachyon::zk::plonk::ColumnType>(kind)) {
          case tachyon::zk::plonk::ColumnType::kAdvice:
            return tachyon::zk::plonk::AdviceColumnKey(index, phase);
          case tachyon::zk::plonk::ColumnType::kInstance:
            return tachyon::zk::plonk::InstanceColumnKey(index);
          case tachyon::zk::plonk::ColumnType::kFixed:
            return tachyon::zk::plonk::FixedColumnKey(index);
          case tachyon::zk::plonk::ColumnType::kAny:
            break;
        }
        NOTREACHED();
        return tachyon::zk::plonk::AnyColumnKey();
      }
    }
  }
};

template <>
class BufferReader<tachyon::zk::plonk::VirtualCell> {
 public:
  static tachyon::zk::plonk::VirtualCell Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    tachyon::zk::plonk::AnyColumnKey column =
        BufferReader<tachyon::zk::plonk::AnyColumnKey>::Read(buffer);
    tachyon::zk::Rotation rotation =
        BufferReader<tachyon::zk::Rotation>::Read(buffer);
    return {column, rotation};
  }
};

template <tachyon::zk::plonk::ColumnType C>
class BufferReader<tachyon::zk::plonk::QueryData<C>> {
 public:
  static tachyon::zk::plonk::QueryData<C> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    if constexpr (C == tachyon::zk::plonk::ColumnType::kAny) {
      NOTREACHED();
    } else {
      tachyon::zk::plonk::ColumnKey<C> column =
          BufferReader<tachyon::zk::plonk::ColumnKey<C>>::Read(buffer);
      tachyon::zk::Rotation rotation =
          BufferReader<tachyon::zk::Rotation>::Read(buffer);
      return {rotation, column};
    }
  }
};

template <tachyon::zk::plonk::ColumnType C>
class BufferReader<tachyon::zk::plonk::Query<C>> {
 public:
  static tachyon::zk::plonk::Query<C> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    bool has_index = ReadU8AsBool(buffer);
    size_t index = has_index ? ReadU32AsSizeT(buffer) : 0;
    size_t column_index = ReadU32AsSizeT(buffer);
    tachyon::zk::Rotation rotation =
        BufferReader<tachyon::zk::Rotation>::Read(buffer);
    if constexpr (C == tachyon::zk::plonk::ColumnType::kAdvice) {
      tachyon::zk::plonk::Phase phase =
          BufferReader<tachyon::zk::plonk::Phase>::Read(buffer);
      return {
          index,
          rotation,
          tachyon::zk::plonk::ColumnKey<C>(column_index, phase),
      };
    } else {
      if constexpr (C == tachyon::zk::plonk::ColumnType::kInstance ||
                    C == tachyon::zk::plonk::ColumnType::kFixed) {
        return {
            index,
            rotation,
            tachyon::zk::plonk::ColumnKey<C>(column_index),
        };
      } else {
        NOTREACHED();
      }
    }
  }
};

template <typename F>
class BufferReader<std::unique_ptr<tachyon::zk::Expression<F>>> {
 public:
  static std::unique_ptr<tachyon::zk::Expression<F>> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    uint8_t kind = BufferReader<uint8_t>::Read(buffer);
    // NOTE(batzor): this switch statement is hardcoded to be compliant with
    // halo2 rust implementation.
    // https://github.com/kroma-network/halo2/blob/4ad135/halo2_proofs/src/plonk/circuit.rs#L993
    switch (kind) {
      case 0:
        return tachyon::zk::ExpressionFactory<F>::Constant(
            BufferReader<F>::Read(buffer));
      case 1:
        return tachyon::zk::plonk::ExpressionFactory<F>::Selector(
            BufferReader<tachyon::zk::plonk::Selector>::Read(buffer));
      case 2:
        return tachyon::zk::plonk::ExpressionFactory<F>::Fixed(
            BufferReader<tachyon::zk::plonk::FixedQuery>::Read(buffer));
      case 3:
        return tachyon::zk::plonk::ExpressionFactory<F>::Advice(
            BufferReader<tachyon::zk::plonk::AdviceQuery>::Read(buffer));
      case 4:
        return tachyon::zk::plonk::ExpressionFactory<F>::Instance(
            BufferReader<tachyon::zk::plonk::InstanceQuery>::Read(buffer));
      case 5:
        return tachyon::zk::plonk::ExpressionFactory<F>::Challenge(
            BufferReader<tachyon::zk::plonk::Challenge>::Read(buffer));
      case 6:
        return tachyon::zk::ExpressionFactory<F>::Negated(
            BufferReader<std::unique_ptr<tachyon::zk::Expression<F>>>::Read(
                buffer));
      case 7: {
        std::unique_ptr<tachyon::zk::Expression<F>> left;
        ReadBuffer(buffer, left);
        std::unique_ptr<tachyon::zk::Expression<F>> right;
        ReadBuffer(buffer, right);
        return tachyon::zk::ExpressionFactory<F>::Sum(std::move(left),
                                                      std::move(right));
      }
      case 8: {
        std::unique_ptr<tachyon::zk::Expression<F>> left;
        ReadBuffer(buffer, left);
        std::unique_ptr<tachyon::zk::Expression<F>> right;
        ReadBuffer(buffer, right);
        return tachyon::zk::ExpressionFactory<F>::Product(std::move(left),
                                                          std::move(right));
      }
      case 9: {
        std::unique_ptr<tachyon::zk::Expression<F>> expr;
        ReadBuffer(buffer, expr);
        F scale = BufferReader<F>::Read(buffer);
        return tachyon::zk::ExpressionFactory<F>::Scaled(std::move(expr),
                                                         std::move(scale));
      }
    }
    NOTREACHED();
    return nullptr;
  }
};

template <typename F>
class BufferReader<tachyon::zk::plonk::Gate<F>> {
 public:
  static tachyon::zk::plonk::Gate<F> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> polys;
    ReadBuffer(buffer, polys);
    std::vector<tachyon::zk::plonk::Selector> queried_selectors;
    ReadBuffer(buffer, queried_selectors);
    std::vector<tachyon::zk::plonk::VirtualCell> queried_cells;
    ReadBuffer(buffer, queried_cells);
    return tachyon::zk::plonk::Gate<F>("", {}, std::move(polys),
                                       std::move(queried_selectors),
                                       std::move(queried_cells));
  }
};

template <>
class BufferReader<tachyon::zk::plonk::PermutationArgument> {
 public:
  static tachyon::zk::plonk::PermutationArgument Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::vector<tachyon::zk::plonk::AnyColumnKey> column_keys;
    ReadBuffer(buffer, column_keys);
    return tachyon::zk::plonk::PermutationArgument(std::move(column_keys));
  }
};

template <typename F>
class BufferReader<tachyon::zk::lookup::Argument<F>> {
 public:
  static tachyon::zk::lookup::Argument<F> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::vector<std::vector<std::unique_ptr<tachyon::zk::Expression<F>>>>
        inputs_expressions;
    ReadBuffer(buffer, inputs_expressions);
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> table_expressions;
    ReadBuffer(buffer, table_expressions);
    return tachyon::zk::lookup::Argument<F>("", std::move(inputs_expressions),
                                            std::move(table_expressions));
  }
};

template <typename F>
class BufferReader<tachyon::zk::plonk::LookupTracker<F>> {
 public:
  static tachyon::zk::plonk::LookupTracker<F> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> table;
    ReadBuffer(buffer, table);
    std::vector<std::vector<std::unique_ptr<tachyon::zk::Expression<F>>>>
        inputs;
    ReadBuffer(buffer, inputs);
    return tachyon::zk::plonk::LookupTracker<F>("", std::move(table),
                                                std::move(inputs));
  }
};

template <typename F>
class BufferReader<tachyon::zk::shuffle::Argument<F>> {
 public:
  static tachyon::zk::shuffle::Argument<F> Read(
      const tachyon::base::ReadOnlyBuffer& buffer) {
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> input_expressions;
    ReadBuffer(buffer, input_expressions);
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>>
        shuffle_expressions;
    ReadBuffer(buffer, shuffle_expressions);
    return tachyon::zk::shuffle::Argument<F>("", std::move(input_expressions),
                                             std::move(shuffle_expressions));
  }
};

template <typename T>
class BufferReader<
    T, std::enable_if_t<
           std::is_same_v<T, c::zk::plonk::halo2::bn254::GWCPCS> ||
           std::is_same_v<T, c::zk::plonk::halo2::bn254::SHPlonkPCS>>> {
 public:
  static T Read(const tachyon::base::ReadOnlyBuffer& buffer) {
    using G1Point = typename T::G1Point;
    using G2Point = typename T::G2Point;

    uint32_t k;
    CHECK(buffer.Read(&k));
    size_t n = size_t{1} << k;
    std::vector<G1Point> g1_powers_of_tau =
        tachyon::base::CreateVector(n, [&buffer]() {
          G1Point point;
          ReadBuffer(buffer, point);
          return point;
        });
    std::vector<G1Point> g1_powers_of_tau_lagrange =
        tachyon::base::CreateVector(n, [&buffer]() {
          G1Point point;
          ReadBuffer(buffer, point);
          return point;
        });

    // NOTE(dongchangYoo): read |g2| but do not use it.
    G2Point g2;
    ReadBuffer(buffer, g2);

    G2Point s_g2;
    ReadBuffer(buffer, s_g2);
    return {std::move(g1_powers_of_tau), std::move(g1_powers_of_tau_lagrange),
            std::move(s_g2)};
  }
};

}  // namespace tachyon::c::zk::plonk

#endif  // TACHYON_C_ZK_PLONK_HALO2_BUFFER_READER_H_
