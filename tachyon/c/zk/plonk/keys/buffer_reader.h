#ifndef TACHYON_C_ZK_PLONK_KEYS_BUFFER_READER_H_
#define TACHYON_C_ZK_PLONK_KEYS_BUFFER_READER_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/zk/plonk/keys/buffer_reader.h"
#include "tachyon/c/zk/plonk/keys/endian_auto_reset.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/base/phase.h"
#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

namespace tachyon::c::zk::plonk {

template <typename T, typename SFINAE = void>
class BufferReader;

template <typename T>
void ReadBuffer(const base::ReadOnlyBuffer& buffer, T& value) {
  value = BufferReader<T>::Read(buffer);
}

template <typename T>
class BufferReader<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  static T Read(const base::ReadOnlyBuffer& buffer) {
    EndianAutoReset resetter(buffer, base::Endian::kBig);
    T v;
    CHECK(buffer.Read(&v));
    return v;
  }
};

inline bool ReadU8AsBool(const base::ReadOnlyBuffer& buffer) {
  return BufferReader<uint8_t>::Read(buffer) != 0;
}

inline size_t ReadU32AsSizeT(const base::ReadOnlyBuffer& buffer) {
  return size_t{BufferReader<uint32_t>::Read(buffer)};
}

template <typename T>
class BufferReader<std::vector<T>> {
 public:
  static std::vector<T> Read(const base::ReadOnlyBuffer& buffer) {
    return base::CreateVector(ReadU32AsSizeT(buffer), [&buffer]() {
      return BufferReader<T>::Read(buffer);
    });
  }
};

template <typename Curve>
class BufferReader<tachyon::math::AffinePoint<Curve>> {
 public:
  using BaseField = typename tachyon::math::AffinePoint<Curve>::BaseField;

  static tachyon::math::AffinePoint<Curve> Read(
      const base::ReadOnlyBuffer& buffer) {
    BaseField x = BufferReader<BaseField>::Read(buffer);
    BaseField y = BufferReader<BaseField>::Read(buffer);
    if (x.IsZero() && y.IsZero()) {
      return {};
    }
    return {std::move(x), std::move(y)};
  }
};

template <typename T>
class BufferReader<
    T,
    std::enable_if_t<std::is_base_of_v<tachyon::math::PrimeFieldBase<T>, T>>> {
 public:
  using BigInt = typename T::BigIntTy;

  static T Read(const base::ReadOnlyBuffer& buffer) {
    EndianAutoReset resetter(buffer, base::Endian::kLittle);
    BigInt montgomery;
    CHECK(buffer.Read(montgomery.limbs));
    return T::FromMontgomery(montgomery);
  }
};

template <typename F, size_t MaxDegree>
class BufferReader<tachyon::math::UnivariateDensePolynomial<F, MaxDegree>> {
 public:
  static tachyon::math::UnivariateDensePolynomial<F, MaxDegree> Read(
      const base::ReadOnlyBuffer& buffer) {
    std::vector<F> coeffs;
    ReadBuffer(buffer, coeffs);
    return tachyon::math::UnivariateDensePolynomial<F, MaxDegree>(
        tachyon::math::UnivariateDenseCoefficients<F, MaxDegree>(
            std::move(coeffs)));
  }
};  // namespace tachyon::c::zk

template <typename F, size_t MaxDegree>
class BufferReader<tachyon::math::UnivariateEvaluations<F, MaxDegree>> {
 public:
  static tachyon::math::UnivariateEvaluations<F, MaxDegree> Read(
      const base::ReadOnlyBuffer& buffer) {
    std::vector<F> evals;
    ReadBuffer(buffer, evals);
    return tachyon::math::UnivariateEvaluations<F, MaxDegree>(std::move(evals));
  }
};  // namespace tachyon::c::zk

template <>
class BufferReader<tachyon::zk::plonk::Phase> {
 public:
  static tachyon::zk::plonk::Phase Read(const base::ReadOnlyBuffer& buffer) {
    return tachyon::zk::plonk::Phase(BufferReader<uint8_t>::Read(buffer));
  }
};

template <>
class BufferReader<tachyon::zk::plonk::Challenge> {
 public:
  static tachyon::zk::plonk::Challenge Read(
      const base::ReadOnlyBuffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    tachyon::zk::plonk::Phase phase =
        BufferReader<tachyon::zk::plonk::Phase>::Read(buffer);
    return {index, phase};
  }
};

template <>
class BufferReader<tachyon::zk::Rotation> {
 public:
  static tachyon::zk::Rotation Read(const base::ReadOnlyBuffer& buffer) {
    return tachyon::zk::Rotation(BufferReader<int32_t>::Read(buffer));
  }
};

template <>
class BufferReader<tachyon::zk::plonk::Selector> {
 public:
  static tachyon::zk::plonk::Selector Read(const base::ReadOnlyBuffer& buffer) {
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
      const base::ReadOnlyBuffer& buffer) {
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
      const base::ReadOnlyBuffer& buffer) {
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
      const base::ReadOnlyBuffer& buffer) {
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
  static tachyon::zk::plonk::Query<C> Read(const base::ReadOnlyBuffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
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
      const base::ReadOnlyBuffer& buffer) {
    uint8_t kind = BufferReader<uint8_t>::Read(buffer);
    switch (kind) {
      case 0:
        return tachyon::zk::ExpressionFactory<F>::Constant(
            BufferReader<F>::Read(buffer));
      case 1:
        return tachyon::zk::ExpressionFactory<F>::Selector(
            BufferReader<tachyon::zk::plonk::Selector>::Read(buffer));
      case 2:
        return tachyon::zk::ExpressionFactory<F>::Fixed(
            BufferReader<tachyon::zk::plonk::FixedQuery>::Read(buffer));
      case 3:
        return tachyon::zk::ExpressionFactory<F>::Advice(
            BufferReader<tachyon::zk::plonk::AdviceQuery>::Read(buffer));
      case 4:
        return tachyon::zk::ExpressionFactory<F>::Instance(
            BufferReader<tachyon::zk::plonk::InstanceQuery>::Read(buffer));
      case 5:
        return tachyon::zk::ExpressionFactory<F>::Challenge(
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
  static tachyon::zk::plonk::Gate<F> Read(const base::ReadOnlyBuffer& buffer) {
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
      const base::ReadOnlyBuffer& buffer) {
    std::vector<tachyon::zk::plonk::AnyColumnKey> column_keys;
    ReadBuffer(buffer, column_keys);
    return tachyon::zk::plonk::PermutationArgument(std::move(column_keys));
  }
};

template <typename F>
class BufferReader<tachyon::zk::LookupArgument<F>> {
 public:
  static tachyon::zk::LookupArgument<F> Read(
      const base::ReadOnlyBuffer& buffer) {
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> input_expressions;
    ReadBuffer(buffer, input_expressions);
    std::vector<std::unique_ptr<tachyon::zk::Expression<F>>> table_expressions;
    ReadBuffer(buffer, table_expressions);
    return tachyon::zk::LookupArgument<F>("", std::move(input_expressions),
                                          std::move(table_expressions));
  }
};

template <typename Poly, typename Evals>
class BufferReader<tachyon::zk::plonk::PermutationProvingKey<Poly, Evals>> {
 public:
  static tachyon::zk::plonk::PermutationProvingKey<Poly, Evals> Read(
      const base::ReadOnlyBuffer& buffer) {
    std::vector<Evals> permutations;
    ReadBuffer(buffer, permutations);
    std::vector<Poly> polys;
    ReadBuffer(buffer, polys);
    return tachyon::zk::plonk::PermutationProvingKey<Poly, Evals>(
        std::move(permutations), std::move(polys));
  }
};

}  // namespace tachyon::c::zk::plonk

#endif  // TACHYON_C_ZK_PLONK_KEYS_BUFFER_READER_H_
