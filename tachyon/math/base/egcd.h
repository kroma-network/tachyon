#ifndef TACHYON_MATH_BASE_EGCD_H_
#define TACHYON_MATH_BASE_EGCD_H_

namespace tachyon::math {

template <typename T>
struct EGCD {
  struct Result {
    T s;
    T t;
    T r;

    bool IsValid(T x, T y) const { return s * x + t * y == r; }
  };

  constexpr static Result Compute(T x, T y) {
    T r_prev = x, r = y, s_prev = 1, s = 0, t_prev = 0, t = 1;

    while (r != 0) {
      T q = r_prev / r;

      T temp = r;
      r = r_prev - q * r;
      r_prev = temp;

      temp = s;
      s = s_prev - q * s;
      s_prev = temp;

      temp = t;
      t = t_prev - q * t;
      t_prev = temp;
    }

    return {s_prev, t_prev, r_prev};
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_EGCD_H_
