#include "tachyon/math/polynomials/multivariate/multivariate_polynomial.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

const size_t kNumVars = 2;
const size_t kMaxDegree = 5;

using Poly = MultivariateSparsePolynomial<GF7, kMaxDegree>;
using Coeffs = MultivariateSparseCoefficients<GF7, kMaxDegree>;

class MultivariatePolynomialTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7Config::Init(); }

  MultivariatePolynomialTest() {
    // poly0: 2
    polys_.push_back(Poly(Coeffs(1, {
                                        {
                                            {
                                                {
                                                    {0, 0}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(2)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>
    // poly1: 1 * x₀
    polys_.push_back(Poly(Coeffs(1, {
                                        {
                                            {
                                                {
                                                    {0, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>
    // poly2: 1 * x₀ (num_vars = 2)
    polys_.push_back(Poly(Coeffs(2, {
                                        {
                                            {
                                                {
                                                    {0, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>
    // poly3: 1 * x₀ + 1 * x₁
    polys_.push_back(Poly(Coeffs(2, {
                                        {
                                            {
                                                {
                                                    {0, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        },                  // Term
                                        {
                                            {
                                                {
                                                    {1, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>

    // poly4: 1 * x₁ + 1 * x₀
    polys_.push_back(Poly(Coeffs(2, {
                                        {
                                            {
                                                {
                                                    {0, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        },                  // Term
                                        {
                                            {
                                                {
                                                    {1, 1}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>

    // poly5: 1 * x₀² + 2 * x₀x₁ + 3 * x₁² + 4
    polys_.push_back(Poly(Coeffs(2, {
                                        {
                                            {
                                                {
                                                    {}  // Element
                                                }       // vector<Element>
                                            },          // Literal
                                            GF7(4)      // coefficient
                                        },              // Term
                                        {
                                            {
                                                {
                                                    {0, 2}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        },                  // Term
                                        {
                                            {
                                                {
                                                    {0, 1},  // Element
                                                    {1, 1}   // Element
                                                }            // vector<Element>
                                            },               // Literal
                                            GF7(2)           // coefficient
                                        },                   // Term
                                        {
                                            {
                                                {
                                                    {1, 2}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(3)          // coefficient
                                        }                   // Term
                                    })));                   // vector<Term>
    // poly6: 1 * x₀³ + 2 * x₀x₁² + 3 * x₁³ + 4
    polys_.push_back(Poly(Coeffs(2, {
                                        {
                                            {
                                                {
                                                    {1, 0}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(4)          // coefficient
                                        },                  // Term
                                        {
                                            {
                                                {
                                                    {0, 3}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(1)          // coefficient
                                        },                  // Term
                                        {
                                            {
                                                {
                                                    {0, 1},  // Element
                                                    {1, 2}   // Element
                                                }            // vector<Element>
                                            },               // Literal
                                            GF7(2)           // coefficient
                                        },                   // Term
                                        {
                                            {
                                                {
                                                    {1, 3}  // Element
                                                }           // vector<Element>
                                            },              // Literal
                                            GF7(3)          // coefficient
                                        },                  // Term
                                    })));                   // vector<Term>
  }
  MultivariatePolynomialTest(const MultivariatePolynomialTest&) = delete;
  MultivariatePolynomialTest& operator=(const MultivariatePolynomialTest&) =
      delete;
  ~MultivariatePolynomialTest() override = default;

 protected:
  std::vector<Poly> polys_;
};

}  // namespace

TEST_F(MultivariatePolynomialTest, IsZero) {
  EXPECT_TRUE(Poly().IsZero());
  EXPECT_TRUE(Poly::Zero().IsZero());
  Coeffs a = Coeffs(1, {
                           {
                               {
                                   {
                                       {0, 0}  // Element
                                   }           // vector<Element>
                               },              // Literal
                               GF7(0)          // coefficient
                           }                   // Term
                       });                     // vector<Term>
  a.Compact();
  EXPECT_TRUE(Poly(a).IsZero());
  for (size_t i = 0; i < polys_.size(); ++i) {
    EXPECT_FALSE(polys_[i].IsZero());
  }
}

TEST_F(MultivariatePolynomialTest, IsOne) {
  EXPECT_TRUE(Poly::One().IsOne());
  EXPECT_TRUE(Poly(Coeffs(1,
                          {
                              {
                                  {
                                      {
                                          {0, 0}  // Element
                                      }           // vector<Element>
                                  },              // Literal
                                  GF7(1)          // coefficient
                              }                   // Term
                          }))                     // vector<Term>
                  .IsOne());
  for (size_t i = 0; i < polys_.size() - 1; ++i) {
    EXPECT_FALSE(polys_[i].IsOne());
  }
}

TEST_F(MultivariatePolynomialTest, Random) {
  bool success = false;
  Poly r = Poly::Random(kNumVars, kMaxDegree);
  for (size_t i = 0; i < 100; ++i) {
    if (r != Poly::Random(kNumVars, kMaxDegree)) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(MultivariatePolynomialTest, IndexingOperator) {
  struct {
    const Poly& poly;
    Coeffs::Literal literal;
    GF7 expected;
  } tests[] = {
      {polys_[0], {{{}}}, GF7(2)},
      {polys_[1], {{{0, 1}}}, GF7(1)},
      {polys_[2], {{{0, 1}}}, GF7(1)},
      {polys_[3], {{{0, 1}}}, GF7(1)},
      {polys_[3], {{{1, 1}}}, GF7(1)},
      {polys_[4], {{{0, 1}}}, GF7(1)},
      {polys_[4], {{{1, 1}}}, GF7(1)},
      {polys_[5], {{{0, 2}}}, GF7(1)},
      {polys_[5], {{{0, 1}, {1, 1}}}, GF7(2)},
      {polys_[5], {{{1, 2}}}, GF7(3)},
      {polys_[5], {{{}}}, GF7(4)},
      {polys_[6], {{{0, 3}}}, GF7(1)},
      {polys_[6], {{{0, 1}, {1, 2}}}, GF7(2)},
      {polys_[6], {{{1, 3}}}, GF7(3)},
      {polys_[6], {{{}}}, GF7(4)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(*test.poly[test.literal], test.expected);
  }
}

TEST_F(MultivariatePolynomialTest, Degree) {
  struct {
    const Poly& poly;
    size_t degree;
  } tests[] = {{polys_[0], 0}, {polys_[1], 1}, {polys_[2], 1}, {polys_[3], 1},
               {polys_[4], 1}, {polys_[5], 2}, {polys_[6], 3}};

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Degree(), test.degree);
  }
  EXPECT_LE(Poly::Random(kNumVars, kMaxDegree).Degree(), kMaxDegree);
}

TEST_F(MultivariatePolynomialTest, Evaluate) {
  struct {
    const Poly& poly;
    std::vector<GF7> evaluation_point;
    GF7 expected;
  } tests[] = {
      {polys_[0], {}, GF7(2)},
      {polys_[1], {GF7(3)}, GF7(3)},
      {polys_[2], {GF7(3)}, GF7(3)},
      {polys_[3], {GF7(3), GF7(3)}, GF7(6)},
      {polys_[4], {GF7(3), GF7(3)}, GF7(6)},
      {polys_[5], {GF7(3), GF7(3)}, GF7(2)},
      {polys_[6], {GF7(3), GF7(3)}, GF7(5)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.Evaluate(test.evaluation_point), test.expected);
  }
}

TEST_F(MultivariatePolynomialTest, ToString) {
  struct {
    const Poly& poly;
    std::string_view expected;
  } tests[] = {
      {polys_[0], "2"},
      {polys_[1], "1 * x_0"},
      {polys_[2], "1 * x_0"},
      {polys_[3], "1 * x_1 + 1 * x_0"},
      {polys_[4], "1 * x_1 + 1 * x_0"},
      {polys_[5], "3 * x_1^2 + 2 * x_1x_0 + 1 * x_0^2 + 4"},
      {polys_[6], "3 * x_1^3 + 2 * x_1^2x_0 + 1 * x_0^3 + 4"},
  };

  EXPECT_EQ(Poly().ToString(), "");

  for (const auto& test : tests) {
    EXPECT_EQ(test.poly.ToString(), test.expected);
  }
}

TEST_F(MultivariatePolynomialTest, AdditiveOperators) {
  struct {
    const Poly& a;
    const Poly& b;
    Poly sum;
    Poly amb;
    Poly bma;
  } tests[] = {
      {
          polys_[0], polys_[1],
          Poly(Coeffs(1,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(2)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>
          Poly(Coeffs(1,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(2)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(6)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>
          Poly(Coeffs(1,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(5)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>
      },
      {
          polys_[0], polys_[3],
          Poly(Coeffs(2,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(2)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {1, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>
          Poly(Coeffs(2,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(2)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(6)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {1, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(6)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>

          Poly(Coeffs(2,
                      {
                          {
                              {
                                  {
                                      {0, 0}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(5)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {0, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          },                  // Term
                          {
                              {
                                  {
                                      {1, 1}  // Element
                                  }           // vector<Element>
                              },              // Literal
                              GF7(1)          // coefficient
                          }                   // Term
                      })),                    // vector<Term>
      },
      {polys_[5], polys_[6],
       Poly(Coeffs(2,
                   {
                       {
                           {
                               {
                                   {1, 0}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(1)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(1)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(1)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 1}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(2)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(3)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 2}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(2)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(3)          // coefficient
                       },                  // Term
                   })),                    // vector<Term>
       Poly(Coeffs(2,
                   {
                       {
                           {
                               {
                                   {0, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(1)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(6)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 1}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(2)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(3)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 2}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(5)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(4)          // coefficient
                       },                  // Term
                   })),                    // vector<Term>

       Poly(Coeffs(2,
                   {
                       {
                           {
                               {
                                   {0, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(6)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(1)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 1}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(5)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 2}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(4)          // coefficient
                       },                  // Term
                       {
                           {
                               {
                                   {0, 1},  // Element
                                   {1, 2}   // Element
                               }            // vector<Element>
                           },               // Literal
                           GF7(2)           // coefficient
                       },                   // Term
                       {
                           {
                               {
                                   {1, 3}  // Element
                               }           // vector<Element>
                           },              // Literal
                           GF7(3)          // coefficient
                       },                  // Term
                   }))},                   // vector<Term>
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    Poly tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}
}  // namespace tachyon::math
