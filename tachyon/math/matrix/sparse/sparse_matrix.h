#ifndef TACHYON_MATH_MATRIX_SPARSE_SPARSE_MATRIX_H_
#define TACHYON_MATH_MATRIX_SPARSE_SPARSE_MATRIX_H_

#include "absl/types/span.h"
#include "third_party/eigen3/Eigen/SparseCore"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::math {

template <typename T>
class CSRSparseMatrix;

template <typename T>
class ELLSparseMatrix {
 public:
  struct Element {
    size_t index;
    T value;

    bool operator<(const Element& other) const { return index < other.index; }
    bool operator==(const Element& other) const {
      return index == other.index && value == other.value;
    }
    bool operator!=(const Element& other) const { return !operator==(other); }
  };

  using Elements = std::vector<Element>;

  ELLSparseMatrix() = default;
  ELLSparseMatrix(const std::vector<Elements>& elements_list)
      : elements_list_(elements_list) {}
  ELLSparseMatrix(std::vector<Elements>&& elements_list)
      : elements_list_(std::move(elements_list)) {}
  ELLSparseMatrix(const ELLSparseMatrix& other) = default;
  ELLSparseMatrix& operator=(const ELLSparseMatrix& other) = default;
  ELLSparseMatrix(ELLSparseMatrix&& other) = default;
  ELLSparseMatrix& operator=(ELLSparseMatrix&& other) = default;

  template <int Options, typename StorageIndex,
            typename InnerIterator = typename Eigen::SparseMatrix<
                T, Options, StorageIndex>::InnerIterator>
  static ELLSparseMatrix FromEigenSparseMatrix(
      const Eigen::SparseMatrix<T, Options, StorageIndex>& matrix) {
    ELLSparseMatrix ret;
    ret.elements_list_.resize(matrix.rows());
    for (int k = 0; k < matrix.outerSize(); ++k) {
      for (InnerIterator it(matrix, k); it; ++it) {
        ret.elements_list_[it.row()].push_back(
            {static_cast<size_t>(it.col()), it.value()});
      }
    }
    ret.Sort();
    return ret;
  }

  static ELLSparseMatrix FromCSR(const CSRSparseMatrix<T>& matrix) {
    return matrix.ToELL();
  }

  std::vector<std::vector<T>> GetData() const {
    return base::Map(elements_list_, [](const Elements& elements) {
      return base::Map(elements,
                       [](const Element& element) { return element.value; });
    });
  }

  std::vector<std::vector<size_t>> GetColumnIndices() const {
    return base::Map(elements_list_, [](const Elements& elements) {
      return base::Map(elements,
                       [](const Element& element) { return element.index; });
    });
  }

  size_t MaxRows() const { return elements_list_.size(); }

  // NOTE: Returns a valid value when |this| is sorted.
  size_t MaxCols() const {
    auto it = std::max_element(elements_list_.begin(), elements_list_.end(),
                               [](const Elements& a, const Elements& b) {
                                 return a.back().index < b.back().index;
                               });
    return it->back().index + 1;
  }

  size_t NonZeros() const {
    return std::accumulate(elements_list_.begin(), elements_list_.end(), 0,
                           [](size_t total, const Elements& elements) {
                             return total + elements.size();
                           });
  }

  // NOTE: Returns a valid value when |this| and |other| are sorted.
  bool operator==(const ELLSparseMatrix& other) const {
    return elements_list_ == other.elements_list_;
  }
  bool operator!=(const ELLSparseMatrix& other) const {
    return !operator==(other);
  }

  void Sort() {
    for (Elements& elements : elements_list_) {
      base::ranges::sort(elements.begin(), elements.end());
    }
  }

  bool IsSorted() const {
    for (const Elements& elements : elements_list_) {
      if (!base::ranges::is_sorted(elements.begin(), elements.end()))
        return false;
    }
    return true;
  }

  // NOTE: Returns a valid value when |this| is sorted.
  CSRSparseMatrix<T> ToCSR() const {
    CSRSparseMatrix<T> ret;
    ret.row_ptrs_.reserve(elements_list_.size() + 1);
    ret.row_ptrs_.push_back(0);
    for (const Elements& elements : elements_list_) {
      ret.row_ptrs_.push_back(ret.row_ptrs_.back() + elements.size());
      for (const Element& element : elements) {
        ret.elements_.push_back({element.index, element.value});
      }
    }
    return ret;
  }

  Eigen::SparseMatrix<T> ToEigenSparseMatrix() const {
    using StorageIndex = typename Eigen::SparseMatrix<T>::StorageIndex;
    std::vector<Eigen::Triplet<T>> coefficients;
    for (size_t i = 0; i < elements_list_.size(); ++i) {
      for (const Element& element : elements_list_[i]) {
        coefficients.push_back({static_cast<StorageIndex>(i),
                                static_cast<StorageIndex>(element.index),
                                element.value});
      }
    }
    Eigen::SparseMatrix<T> ret(MaxRows(), MaxCols());
    ret.setFromTriplets(coefficients.begin(), coefficients.end());
    return ret;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "  data: " << base::Vector2DToString(GetData()) << "\n";
    ss << "  col_indices: " << base::Vector2DToString(GetColumnIndices())
       << "\n";
    ss << "}";
    return ss.str();
  }

 private:
  friend class CSRSparseMatrix<T>;

  std::vector<Elements> elements_list_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ELLSparseMatrix<T>& matrix) {
  return os << matrix.ToString();
}

template <typename T>
class CSRSparseMatrix {
 public:
  struct Element {
    size_t index;
    T value;

    bool operator<(const Element& other) const { return index < other.index; }
    bool operator==(const Element& other) const {
      return index == other.index && value == other.value;
    }
    bool operator!=(const Element& other) const { return !operator==(other); }
  };

  using Elements = std::vector<Element>;

  CSRSparseMatrix() = default;
  CSRSparseMatrix(const Elements& elements, const std::vector<size_t>& row_ptrs)
      : elements_(elements), row_ptrs_(row_ptrs) {}
  CSRSparseMatrix(Elements&& elements, std::vector<size_t>&& row_ptrs)
      : elements_(std::move(elements)), row_ptrs_(std::move(row_ptrs)) {}
  CSRSparseMatrix(const CSRSparseMatrix& other) = default;
  CSRSparseMatrix& operator=(const CSRSparseMatrix& other) = default;
  CSRSparseMatrix(CSRSparseMatrix&& other) = default;
  CSRSparseMatrix& operator=(CSRSparseMatrix&& other) = default;

  template <int Options, typename StorageIndex,
            typename InnerIterator = typename Eigen::SparseMatrix<
                T, Options, StorageIndex>::InnerIterator>
  static CSRSparseMatrix FromEigenSparseMatrix(
      const Eigen::SparseMatrix<T, Options, StorageIndex>& matrix) {
    return FromELL(ELLSparseMatrix<T>::FromEigenSparseMatrix(matrix));
  }

  static CSRSparseMatrix FromELL(const ELLSparseMatrix<T>& matrix) {
    return matrix.ToCSR();
  }

  const std::vector<size_t>& row_ptrs() const { return row_ptrs_; }

  std::vector<T> GetData() const {
    return base::Map(elements_,
                     [](const Element& element) { return element.value; });
  }

  std::vector<size_t> GetColumnIndices() const {
    return base::Map(elements_,
                     [](const Element& element) { return element.index; });
  }

  size_t MaxRows() const {
    if (row_ptrs_.empty()) return 0;
    return row_ptrs_.size() - 1;
  }

  // NOTE: Returns a valid value when |this| is sorted.
  size_t MaxCols() const {
    auto it = std::max_element(
        elements_.begin(), elements_.end(),
        [](const Element& a, const Element& b) { return a.index < b.index; });
    return it->index + 1;
  }

  size_t NonZeros() const { return elements_.size(); }

  // Returns a valid value when |this| and |other| are sorted.
  bool operator==(const CSRSparseMatrix& other) const {
    return elements_ == other.elements_ && row_ptrs_ == other.row_ptrs_;
  }
  bool operator!=(const CSRSparseMatrix& other) const {
    return !operator==(other);
  }

  void Sort() {
    for (size_t i = 0; i < row_ptrs_.size(); ++i) {
      if (i != row_ptrs_.size() - 1) {
        base::ranges::sort(elements_.begin() + row_ptrs_[i],
                           elements_.begin() + row_ptrs_[i + 1]);
      }
    }
    base::ranges::sort(row_ptrs_.begin(), row_ptrs_.end());
  }

  bool IsSorted() const {
    for (size_t i = 0; i < row_ptrs_.size(); ++i) {
      if (i != row_ptrs_.size() - 1) {
        if (!base::ranges::is_sorted(elements_.begin() + row_ptrs_[i],
                                     elements_.begin() + row_ptrs_[i + 1]))
          return false;
      }
    }
    return base::ranges::is_sorted(row_ptrs_);
  }

  // NOTE: Returns a valid value when |this| is sorted.
  ELLSparseMatrix<T> ToELL() const {
    using ELLElements = typename ELLSparseMatrix<T>::Elements;
    ELLSparseMatrix<T> ret;
    auto it = elements_.begin();
    for (size_t i = 0; i < row_ptrs_.size(); ++i) {
      if (i != row_ptrs_.size() - 1) {
        size_t cols = row_ptrs_[i + 1] - row_ptrs_[i];
        ELLElements elements;
        elements.reserve(cols);
        for (size_t j = 0; j < cols; ++j, ++it) {
          elements.push_back({it->index, it->value});
        }
        ret.elements_list_.push_back(std::move(elements));
      }
    }
    return ret;
  }

  Eigen::SparseMatrix<T> ToEigenSparseMatrix() const {
    using StorageIndex = typename Eigen::SparseMatrix<T>::StorageIndex;
    std::vector<Eigen::Triplet<T>> coefficients;
    auto it = elements_.begin();
    for (size_t i = 0; i < row_ptrs_.size(); ++i) {
      if (i != row_ptrs_.size() - 1) {
        size_t cols = row_ptrs_[i + 1] - row_ptrs_[i];
        for (size_t j = 0; j < cols; ++j, ++it) {
          coefficients.push_back({static_cast<StorageIndex>(i),
                                  static_cast<StorageIndex>(it->index),
                                  it->value});
        }
      }
    }
    Eigen::SparseMatrix<T> ret(MaxRows(), MaxCols());
    ret.setFromTriplets(coefficients.begin(), coefficients.end());
    return ret;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "  data: " << base::VectorToString(GetData()) << "\n";
    ss << "  col_indices: " << base::VectorToString(GetColumnIndices()) << "\n";
    ss << "  row_ptrs: " << base::VectorToString(row_ptrs_) << "\n";
    ss << "}";
    return ss.str();
  }

 private:
  friend class ELLSparseMatrix<T>;

  Elements elements_;
  std::vector<size_t> row_ptrs_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const CSRSparseMatrix<T>& matrix) {
  return os << matrix.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_SPARSE_SPARSE_MATRIX_H_
