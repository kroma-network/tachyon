#include "tachyon/zk/air/plonky3/base/two_adic_multiplicative_coset.h"

#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::air::plonky3 {
namespace {

using F = math::BabyBear;
using ExtField = math::BabyBear4;

class TwoAdicMultiplicativeCosetTest : public math::FiniteFieldTest<ExtField> {
 public:
  void SetUp() override {
    p0_ = ExtField(F(11) * F(64));
    mult_coset_ = TwoAdicMultiplicativeCoset<F>(5, F(11));
  }

 protected:
  ExtField p0_;
  TwoAdicMultiplicativeCoset<F> mult_coset_;
};

}  // namespace

// TODO(ashjeong): Generalize all tests.
TEST_F(TwoAdicMultiplicativeCosetTest, GetNextPoint) {
  ExtField res = mult_coset_.GetNextPoint(p0_);
  EXPECT_EQ(res, ExtField(F(1528649335)));
}

TEST_F(TwoAdicMultiplicativeCosetTest, CreateDisjointDomain) {
  TwoAdicMultiplicativeCoset<F> disjoint_domain =
      mult_coset_.CreateDisjointDomain(3);
  EXPECT_EQ(disjoint_domain.domain()->size(), 4);
  EXPECT_EQ(disjoint_domain.domain()->offset(), F(341));
}

TEST_F(TwoAdicMultiplicativeCosetTest, GetZpAtPoint) {
  EXPECT_EQ(mult_coset_.GetZpAtPoint(ExtField(mult_coset_.domain()->offset())),
            ExtField::Zero());
  EXPECT_EQ(mult_coset_.GetZpAtPoint(p0_), ExtField(F(193919811)));
}

TEST_F(TwoAdicMultiplicativeCosetTest, SplitDomains) {
  std::vector<TwoAdicMultiplicativeCoset<F>> split_domains =
      mult_coset_.SplitDomains(4);
  std::vector<F> expected{F(11), F(307000666), F(147092939), F(1567650515)};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(split_domains[i].domain()->size(), 8);
    EXPECT_EQ(split_domains[i].domain()->offset(), expected[i]);
  }
}

TEST_F(TwoAdicMultiplicativeCosetTest, GetSelectorsAtPoint) {
  LagrangeSelectors<ExtField> expected{
      ExtField(F(98947898)), ExtField(F(425615992)), ExtField(F(740045704)),
      ExtField(F(1108440347))};
  EXPECT_EQ(mult_coset_.GetSelectorsAtPoint(p0_), expected);
}

TEST_F(TwoAdicMultiplicativeCosetTest, GetSelectorsOnCoset) {
  TwoAdicMultiplicativeCoset<F> other_coset =
      TwoAdicMultiplicativeCoset<F>(2, F(7));
  mult_coset_ = TwoAdicMultiplicativeCoset<F>(2, F::One());
  LagrangeSelectors<std::vector<F>> result =
      mult_coset_.GetSelectorsOnCoset(other_coset);
  std::vector<F> expected{F(400), F(1089934753), F(2013265621), F(923331072)};
  EXPECT_EQ(result.first_row, expected);
  expected =
      std::vector<F>{F(1593752394), F(901253718), F(1593751722), F(811594297)};
  EXPECT_EQ(result.last_row, expected);
  expected = std::vector<F>{F(1728404520), F(1747640578), F(1728404506),
                            F(1709168448)};
  EXPECT_EQ(result.transition, expected);
  expected = std::vector<F>{F(1609773876), F(1609773876), F(1609773876),
                            F(1609773876)};
  EXPECT_EQ(result.inv_zeroifier, expected);
}

}  // namespace tachyon::zk::air::plonky3
