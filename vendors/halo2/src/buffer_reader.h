#ifndef VENDORS_HALO2_SRC_BUFFER_READER_H_
#define VENDORS_HALO2_SRC_BUFFER_READER_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/expressions/expression_factory.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/gate.h"
#include "tachyon/zk/plonk/circuit/phase.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "vendors/halo2/src/endian_auto_reset.h"

namespace tachyon::halo2_api {

template <typename T, typename SFINAE = void>
class BufferReader;

template <typename T>
void ReadBuffer(base::Buffer& buffer, T& value) {
  value = BufferReader<T>::Read(buffer);
}

template <typename T>
class BufferReader<T, std::enable_if_t<std::is_integral_v<T>>> {
 public:
  static T Read(base::Buffer& buffer) {
    EndianAutoReset resetter(buffer, base::Endian::kBig);
    T v;
    CHECK(buffer.Read(&v));
    return v;
  }
};

bool ReadU8AsBool(base::Buffer& buffer) {
  return BufferReader<uint8_t>::Read(buffer) != 0;
}

size_t ReadU32AsSizeT(base::Buffer& buffer) {
  return size_t{BufferReader<uint32_t>::Read(buffer)};
}

template <typename T>
class BufferReader<std::vector<T>> {
 public:
  static std::vector<T> Read(base::Buffer& buffer) {
    return base::CreateVector(ReadU32AsSizeT(buffer), [&buffer]() {
      return BufferReader<T>::Read(buffer);
    });
  }
};

template <typename Curve>
class BufferReader<math::AffinePoint<Curve>> {
 public:
  using BaseField = typename math::AffinePoint<Curve>::BaseField;

  static math::AffinePoint<Curve> Read(base::Buffer& buffer) {
    BaseField x = BufferReader<BaseField>::Read(buffer);
    BaseField y = BufferReader<BaseField>::Read(buffer);
    return {std::move(x), std::move(y)};
  }
};

template <typename T>
class BufferReader<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>> {
 public:
  using BigInt = typename T::BigIntTy;

  static T Read(base::Buffer& buffer) {
    EndianAutoReset resetter(buffer, base::Endian::kLittle);
    BigInt montgomery;
    CHECK(buffer.Read(montgomery.limbs));
    return T::FromMontgomery(montgomery);
  }
};

template <typename F, size_t MaxDegree>
class BufferReader<math::UnivariateDensePolynomial<F, MaxDegree>> {
 public:
  static math::UnivariateDensePolynomial<F, MaxDegree> Read(
      base::Buffer& buffer) {
    std::vector<F> coeffs;
    ReadBuffer(buffer, coeffs);
    return math::UnivariateDensePolynomial<F, MaxDegree>(
        math::UnivariateDenseCoefficients<F, MaxDegree>(std::move(coeffs)));
  }
};

template <typename F, size_t MaxDegree>
class BufferReader<math::UnivariateEvaluations<F, MaxDegree>> {
 public:
  static math::UnivariateEvaluations<F, MaxDegree> Read(base::Buffer& buffer) {
    std::vector<F> evals;
    ReadBuffer(buffer, evals);
    return math::UnivariateEvaluations<F, MaxDegree>(std::move(evals));
  }
};

template <>
class BufferReader<zk::Phase> {
 public:
  static zk::Phase Read(base::Buffer& buffer) {
    return zk::Phase(BufferReader<uint8_t>::Read(buffer));
  }
};

template <>
class BufferReader<zk::Challenge> {
 public:
  static zk::Challenge Read(base::Buffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    zk::Phase phase = BufferReader<zk::Phase>::Read(buffer);
    return {index, phase};
  }
};

template <>
class BufferReader<zk::Rotation> {
 public:
  static zk::Rotation Read(base::Buffer& buffer) {
    return zk::Rotation(BufferReader<int32_t>::Read(buffer));
  }
};

template <>
class BufferReader<zk::Selector> {
 public:
  static zk::Selector Read(base::Buffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    bool is_simple = ReadU8AsBool(buffer);
    return is_simple ? zk::Selector::Simple(index)
                     : zk::Selector::Complex(index);
  }
};

template <zk::ColumnType C>
class BufferReader<zk::ColumnKey<C>> {
 public:
  static zk::ColumnKey<C> Read(base::Buffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    uint8_t kind = BufferReader<uint8_t>::Read(buffer);
    if constexpr (C == zk::ColumnType::kAdvice) {
      CHECK_EQ(kind, static_cast<int8_t>(C));
      return zk::ColumnKey<C>(index, BufferReader<zk::Phase>::Read(buffer));
    } else {
      if constexpr (C == zk::ColumnType::kInstance ||
                    C == zk::ColumnType::kFixed) {
        CHECK_EQ(kind, static_cast<int8_t>(C));
        return zk::ColumnKey<C>(index);
      } else {
        zk::Phase phase = BufferReader<zk::Phase>::Read(buffer);
        switch (static_cast<zk::ColumnType>(kind)) {
          case zk::ColumnType::kAdvice:
            return zk::AdviceColumnKey(index, phase);
          case zk::ColumnType::kInstance:
            return zk::InstanceColumnKey(index);
          case zk::ColumnType::kFixed:
            return zk::FixedColumnKey(index);
          case zk::ColumnType::kAny:
            break;
        }
        NOTREACHED();
        return zk::AnyColumnKey();
      }
    }
  }
};

template <>
class BufferReader<zk::VirtualCell> {
 public:
  static zk::VirtualCell Read(base::Buffer& buffer) {
    zk::AnyColumnKey column = BufferReader<zk::AnyColumnKey>::Read(buffer);
    zk::Rotation rotation = BufferReader<zk::Rotation>::Read(buffer);
    return {column, rotation};
  }
};

template <zk::ColumnType C>
class BufferReader<zk::QueryData<C>> {
 public:
  static zk::QueryData<C> Read(base::Buffer& buffer) {
    if constexpr (C == zk::ColumnType::kAny) {
      NOTREACHED();
    } else {
      zk::ColumnKey<C> column = BufferReader<zk::ColumnKey<C>>::Read(buffer);
      zk::Rotation rotation = BufferReader<zk::Rotation>::Read(buffer);
      return {rotation, column};
    }
  }
};

template <zk::ColumnType C>
class BufferReader<zk::Query<C>> {
 public:
  static zk::Query<C> Read(base::Buffer& buffer) {
    size_t index = ReadU32AsSizeT(buffer);
    size_t column_index = ReadU32AsSizeT(buffer);
    zk::Rotation rotation = BufferReader<zk::Rotation>::Read(buffer);
    if constexpr (C == zk::ColumnType::kAdvice) {
      zk::Phase phase = BufferReader<zk::Phase>::Read(buffer);
      return {
          index,
          rotation,
          zk::ColumnKey<C>(column_index, phase),
      };
    } else {
      if constexpr (C == zk::ColumnType::kInstance ||
                    C == zk::ColumnType::kFixed) {
        return {
            index,
            rotation,
            zk::ColumnKey<C>(column_index),
        };
      } else {
        NOTREACHED();
      }
    }
  }
};

template <typename F>
class BufferReader<std::unique_ptr<zk::Expression<F>>> {
 public:
  static std::unique_ptr<zk::Expression<F>> Read(base::Buffer& buffer) {
    uint8_t kind = BufferReader<uint8_t>::Read(buffer);
    switch (kind) {
      case 0:
        return zk::ExpressionFactory<F>::Constant(
            BufferReader<F>::Read(buffer));
      case 1:
        return zk::ExpressionFactory<F>::Selector(
            BufferReader<zk::Selector>::Read(buffer));
      case 2:
        return zk::ExpressionFactory<F>::Fixed(
            BufferReader<zk::FixedQuery>::Read(buffer));
      case 3:
        return zk::ExpressionFactory<F>::Advice(
            BufferReader<zk::AdviceQuery>::Read(buffer));
      case 4:
        return zk::ExpressionFactory<F>::Instance(
            BufferReader<zk::InstanceQuery>::Read(buffer));
      case 5:
        return zk::ExpressionFactory<F>::Challenge(
            BufferReader<zk::Challenge>::Read(buffer));
      case 6:
        return zk::ExpressionFactory<F>::Negated(
            BufferReader<std::unique_ptr<zk::Expression<F>>>::Read(buffer));
      case 7: {
        std::unique_ptr<zk::Expression<F>> left;
        ReadBuffer(buffer, left);
        std::unique_ptr<zk::Expression<F>> right;
        ReadBuffer(buffer, right);
        return zk::ExpressionFactory<F>::Sum(std::move(left), std::move(right));
      }
      case 8: {
        std::unique_ptr<zk::Expression<F>> left;
        ReadBuffer(buffer, left);
        std::unique_ptr<zk::Expression<F>> right;
        ReadBuffer(buffer, right);
        return zk::ExpressionFactory<F>::Product(std::move(left),
                                                 std::move(right));
      }
      case 9: {
        std::unique_ptr<zk::Expression<F>> expr;
        ReadBuffer(buffer, expr);
        F scale = BufferReader<F>::Read(buffer);
        return zk::ExpressionFactory<F>::Scaled(std::move(expr),
                                                std::move(scale));
      }
    }
    NOTREACHED();
    return nullptr;
  }
};

template <typename F>
class BufferReader<zk::Gate<F>> {
 public:
  static zk::Gate<F> Read(base::Buffer& buffer) {
    std::vector<std::unique_ptr<zk::Expression<F>>> polys;
    ReadBuffer(buffer, polys);
    std::vector<zk::Selector> queried_selectors;
    ReadBuffer(buffer, queried_selectors);
    std::vector<zk::VirtualCell> queried_cells;
    ReadBuffer(buffer, queried_cells);
    return zk::Gate<F>("", {}, std::move(polys), std::move(queried_selectors),
                       std::move(queried_cells));
  }
};

template <>
class BufferReader<zk::PermutationArgument> {
 public:
  static zk::PermutationArgument Read(base::Buffer& buffer) {
    std::vector<zk::AnyColumnKey> column_keys;
    ReadBuffer(buffer, column_keys);
    return zk::PermutationArgument(std::move(column_keys));
  }
};

template <typename F>
class BufferReader<zk::LookupArgument<F>> {
 public:
  static zk::LookupArgument<F> Read(base::Buffer& buffer) {
    std::vector<std::unique_ptr<zk::Expression<F>>> input_expressions;
    ReadBuffer(buffer, input_expressions);
    std::vector<std::unique_ptr<zk::Expression<F>>> table_expressions;
    ReadBuffer(buffer, table_expressions);
    return zk::LookupArgument<F>("", std::move(input_expressions),
                                 std::move(table_expressions));
  }
};

template <typename Poly, typename Evals>
class BufferReader<zk::PermutationProvingKey<Poly, Evals>> {
 public:
  static zk::PermutationProvingKey<Poly, Evals> Read(base::Buffer& buffer) {
    std::vector<Evals> permutations;
    ReadBuffer(buffer, permutations);
    std::vector<Poly> polys;
    ReadBuffer(buffer, polys);
    return zk::PermutationProvingKey<Poly, Evals>(std::move(permutations),
                                                  std::move(polys));
  }
};

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_BUFFER_READER_H_
