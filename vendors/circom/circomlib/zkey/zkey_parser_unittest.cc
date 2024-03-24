#include "circomlib/zkey/zkey_parser.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::circom {

namespace {

class ZKeyParserTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G2Curve::Init(); }
};

G1AffinePoint ToG1AffinePoint(std::string_view g1[2]) {
  math::bn254::Fq x = math::bn254::Fq::FromDecString(g1[0]);
  math::bn254::Fq y = math::bn254::Fq::FromDecString(g1[1]);
  bool infinity = x.IsZero() && y.IsZero();
  math::bn254::G1AffinePoint affine_point(std::move(x), std::move(y), infinity);
  return G1AffinePoint::FromNative<true>(std::move(affine_point));
}

G2AffinePoint ToG2AffinePoint(std::string_view g2[2][2]) {
  math::bn254::Fq2 x(math::bn254::Fq::FromDecString(g2[0][0]),
                     math::bn254::Fq::FromDecString(g2[0][1]));
  math::bn254::Fq2 y(math::bn254::Fq::FromDecString(g2[1][0]),
                     math::bn254::Fq::FromDecString(g2[1][1]));
  bool infinity = x.IsZero() && y.IsZero();
  math::bn254::G2AffinePoint affine_point(std::move(x), std::move(y), infinity);
  return G2AffinePoint::FromNative<true>(std::move(affine_point));
}

struct CellData {
  std::string_view coefficient;
  uint32_t signal = 0;
};

Cell ToCell(const CellData& data) {
  return {
      PrimeField::FromNative<true>(
          math::bn254::Fr::FromDecString(data.coefficient)),
      data.signal,
  };
}

}  // namespace

TEST_F(ZKeyParserTest, Parse) {
  ZKeyParser parser;
  std::unique_ptr<ZKey> zkey =
      parser.Parse(base::FilePath("examples/multiplier_3.zkey"));
  ASSERT_TRUE(zkey);
  ASSERT_EQ(zkey->GetVersion(), 1);

  v1::ZKey* v1_zkey = zkey->ToV1();
  v1_zkey->Normalize<math::bn254::Fq, math::bn254::Fr>();

  v1::ZKeyHeaderSection expected_header = {1};
  EXPECT_EQ(v1_zkey->header, expected_header);

  std::array<uint8_t, 32> fq_bytes =
      math::bn254::FqConfig::kModulus.ToBytesLE();
  std::array<uint8_t, 32> fr_bytes =
      math::bn254::FrConfig::kModulus.ToBytesLE();

  // clang-format off
  std::string_view alpha_g1[2] = {
    "5700502584084766622350343367608487274977128430049880895783423261700075212785",
    "9143870410831450591509938003078256759736333300521257694515214164265805259830",
  };
  std::string_view beta_g1[2] = {
    "12699714711422499622362310820475830692566951228171954587615996781136226772367",
    "2601999511749500018822665665362525344184434745926911293241192574303473253831",
  };
  std::string_view beta_g2[2][2] = {
    {
      "11780196173848324687642894328871430898972567583635494711927265792805257024861",
      "3029614260803671687015271480824975868088527303860361358764452805565479529001",
    },
    {
      "17817615377642575824268866714659516420384007262298492272608472268977629075434",
      "10565581580493997556536063930500447170628763955833078597453719665182760199848"
    },
  };
  std::string_view gamma_g2[2][2] = {
    {
      "10857046999023057135944570762232829481370756359578518086990519993285655852781",
      "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    },
    {
      "8495653923123431417604973247489272438418190587263600148770280649306958101930",
      "4082367875863433681332203403145435568316851327593401208105741076214120093531"
    },
  };
  std::string_view delta_g1[2] =  {
    "18121096455458648748006856505340317178704791872899059396361359566439114201168",
    "1584219057669659447306711278235088033786171030532185363250775914928871374123",
  };
  std::string_view delta_g2[2][2] = {
    {
      "12202969132968262321607709426694195568617274504067880373401633817598633451917",
      "3518609536734967363313514083213104545803320919611827958997148133314959258666",
    },
    {
      "18833075355260917945174052299300691737626083696241282320643937004218902395077",
      "9205653431456404075288921182558898955030763586613450941955464042050735041036"
    },
  };
  // clang-format on

  v1::ZKeyHeaderGrothSection expected_header_groth = {
      PrimeField{std::vector<uint8_t>(fq_bytes.begin(), fq_bytes.end())},
      PrimeField{std::vector<uint8_t>(fr_bytes.begin(), fr_bytes.end())},
      6,
      1,
      4,
      {
          ToG1AffinePoint(alpha_g1),
          ToG1AffinePoint(beta_g1),
          ToG2AffinePoint(beta_g2),
          ToG2AffinePoint(gamma_g2),
          ToG1AffinePoint(delta_g1),
          ToG2AffinePoint(delta_g2),
      }};
  EXPECT_EQ(v1_zkey->header_groth, expected_header_groth);

  // clang-format off
  std::string_view expected_ic_strs[][2] = {
    {
      "1400989341879513116647759947859271187117391672677487101192308885590924596480",
      "18827163924960691750679623127657074266908067481725903803154895122477837234033",
    },
    {
      "21594466749489205217527764338982336503042047314074010288653645878163201627935",
      "12389944772204356478767528982065842735667468360799583461061670022670871632383",
    }
  };
  // clang-format on
  v1::ICSection expected_ic = {base::Map(
      expected_ic_strs,
      [](std::string_view g1_str[2]) { return ToG1AffinePoint(g1_str); })};
  EXPECT_EQ(v1_zkey->ic, expected_ic);

  // clang-format off
  CellData a[4][1] = {
    {{"15537367993719455909907449462855742678020201736855642022731641111541721333766", 2}},
    {{"15537367993719455909907449462855742678020201736855642022731641111541721333766", 5}},
    {{"6350874878119819312338956282401532410528162663560392320966563075034087161851", 0}},
    {{"6350874878119819312338956282401532410528162663560392320966563075034087161851", 1}},
  };
  CellData b[2][1] = {
    {{"6350874878119819312338956282401532410528162663560392320966563075034087161851", 3}},
    {{"6350874878119819312338956282401532410528162663560392320966563075034087161851", 4}},
  };
  // clang-format on

  v1::CoefficientsSection expected_coefficients;
  expected_coefficients.a = base::Map(a, [](const CellData row[1]) {
    return std::vector<Cell>{ToCell(row[0])};
  });
  expected_coefficients.b = base::Map(b, [](const CellData row[1]) {
    return std::vector<Cell>{ToCell(row[0])};
  });
  expected_coefficients.b.resize(4);
  EXPECT_EQ(v1_zkey->coefficients, expected_coefficients);

  // clang-format off
  std::string_view expected_points_a1_strs[][2] = {
    {
      "8858563469144920540528478490224638442973773873152551307670564100347093499191",
      "7888214391937843930525848128254405915157714572978190674521564636068162216311",
    },
    {
      "14537214592124271965353533016257772100455033778428577041971202446686849252644",
      "2198766467867023896703420308951432042782623727887618971273865174145643356495",
    },
    {
      "8437302598248383817148383036741547214048558400312301295747047351838256772123",
      "4253086419746464003785043685439509391398040483296248505707498714848332192725",
    },
    {
      "0",
      "0",
    },
    {
      "0",
      "0",
    },
    {
      "18141870587741836486360437684811661514896911334995841933942081072546739652377",
      "11898889550822544273094627075076607374273361105699305622414170117806818640166",
    },
  };
  // clang-format on
  v1::PointsA1Section expected_points_a1 = {base::Map(
      expected_points_a1_strs,
      [](std::string_view g1_str[2]) { return ToG1AffinePoint(g1_str); })};
  EXPECT_EQ(v1_zkey->points_a1, expected_points_a1);

  // clang-format off
  std::string_view expected_points_b1_strs[][2] = {
    {
      "0",
      "0",
    },
    {
      "0",
      "0",
    },
    {
      "0",
      "0",
    },
    {
      "8437302598248383817148383036741547214048558400312301295747047351838256772123",
      "17635156452092811218461362059817765697298270674001575156981539179796894015858",
    },
    {
      "18141870587741836486360437684811661514896911334995841933942081072546739652377",
      "9989353321016730949151778670180667714422950051598518040274867776838407568417",
    },
    {
      "0",
      "0",
    },
  };
  // clang-format on
  v1::PointsB1Section expected_points_b1 = {base::Map(
      expected_points_b1_strs,
      [](std::string_view g1_str[2]) { return ToG1AffinePoint(g1_str); })};
  EXPECT_EQ(v1_zkey->points_b1, expected_points_b1);

  // clang-format off
  std::string_view expected_points_b2_strs[][2][2] = {
    {
      {
        "0",
        "0",
      },
      {
        "0",
        "0",
      },
    },
    {
      {
        "0",
        "0",
      },
      {
        "0",
        "0",
      },
    },
    {
      {
        "0",
        "0",
      },
      {
        "0",
        "0",
      },
    },
    {
      {
        "11802355142842158477844840276643950524651646845857959153292587217381565327694",
        "457802140950752837652486610695137856486735570684796633854607481168528542690",
      },
      {
        "20890171563279906566473411100997246109253767120687561732766386895834736963353",
        "16651273739129079927357167917012486078894432847976493185750898267875291365102",
      },
    },
    {
      {
        "10497384263581811947331014280742114350358633905325456855203962488034692371918",
        "18830443503724104126054724199526976415213768169570049829023119362975529483862",
      },
      {
        "2897811547300447900653975040323384022566235978103493011133012417217802784890",
        "16901016173912215869936744693239612773536725857766261524800318448291024785326",
      },
    },
    {
      {
        "0",
        "0",
      },
      {
        "0",
        "0",
      },
    },
  };
  // clang-format on
  v1::PointsB2Section expected_points_b2 = {base::Map(
      expected_points_b2_strs,
      [](std::string_view g2_str[2][2]) { return ToG2AffinePoint(g2_str); })};
  EXPECT_EQ(v1_zkey->points_b2, expected_points_b2);

  // clang-format off
  std::string_view expected_points_c1_strs[][2] = {
    {
      "6484623682921116324480150495004051666793989100055782957602702403603681644087",
      "9929291865986144258563515295092942842219006739962304434423595963929152482511",
    },
    {
      "2428323497801148585616314929247100239345590148163609103036877009446204389916",
      "1369420932117767430108173316323061012699395351388826295491040858566319703347",
    },
    {
      "13708611801147211497962477788407562751837989814094039031176550253520745906453",
      "18151634040197625351146955912693877471406025170780331525940368525891263003135",
    },
    {
      "18284461151151484053721771333035891989697029546383442693996002288554315091425",
      "16368214353654036882062732785064467942017556803959036709508171804841339585293",
    },
  };
  // clang-format on
  v1::PointsC1Section expected_points_c1 = {base::Map(
      expected_points_c1_strs,
      [](std::string_view g1_str[2]) { return ToG1AffinePoint(g1_str); })};
  EXPECT_EQ(v1_zkey->points_c1, expected_points_c1);

  // clang-format off
  std::string_view expected_points_h1_strs[][2] = {
    {
      "14491578304961494983864577860005130648083128241617887000567655109826889517765",
      "747219666158167233177659053461241136834613714217214833065101750371763762297",
    },
    {
      "18146981344323077863601561933187745037217273647724943548414954460732389867565",
      "12817730263653070328990536695886808120537471796272335985833677017167603721628",
    },
    {
      "4366845324910879958335909021287873980685276473035323564144861870453367362588",
      "15317533781337899355741367900150169453079806490427679757823792291054041331892",
    },
    {
      "2710629677537908437939962149843601091625518295461558128309313181856887222076",
      "5586033426926327725348881354712158558954133315341909946743947206034841045334",
    },
  };
  // clang-format on
  v1::PointsH1Section expected_points_h1 = {base::Map(
      expected_points_h1_strs,
      [](std::string_view g1_str[2]) { return ToG1AffinePoint(g1_str); })};
  EXPECT_EQ(v1_zkey->points_h1, expected_points_h1);
}

}  // namespace tachyon::circom
