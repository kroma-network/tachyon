#include "tachyon/math/base/byinverter.h"

#include "gtest/gtest.h"

#include "tachyon/math/base/big_int.h"

namespace tachyon::math {

// The cases of a prime modulus and trivial representation of
// both input and output of the inversion method. The modulus
// is the order of the scalar field of the bn254 curve
TEST(BYInverterTest, PrimeTrivial) {
  BigInt<4> modulus = *BigInt<4>::FromDecString(
      "218882428718392752222464057452572750885483644004160343436982041865758084"
      "95617");
  BigInt<4> adjuster = BigInt<4>::One();
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  input = *BigInt<4>::FromDecString(
      "103016822455395885937009463448678224530860491458983009780242298691299944"
      "91070");
  expected = *BigInt<4>::FromDecString(
      "198619552734958050563237927350841782377548356603726300802008990262189760"
      "35016");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  input = *BigInt<4>::FromDecString(
      "126520449418548447151734230498264169965304739797018972163764030461079803"
      "31138");
  expected = *BigInt<4>::FromDecString(
      "137411891073970879827055042948574283647817819871615941505158449651527160"
      "38798");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  input = *BigInt<4>::FromDecString(
      "212303313650938401560033971399886135220900061766237263322117467937143842"
      "61253");
  expected = *BigInt<4>::FromDecString(
      "199212938989172163056655895547418784289276648223865752630701101392037405"
      "91967");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a prime modulus and Montgomery representation of both
// input and output of the inversion method. The modulus is the order
// of the scalar field of the bn254 curve. The Montgomery factor equals
// 2²⁵⁶.For the numbers specified in Montgomery representation their
// trivial form is in the comments
TEST(BYInverterTest, PrimeMontgomery) {
  BigInt<4> modulus = *BigInt<4>::FromDecString(
      "218882428718392752222464057452572750885483644004160343436982041865758084"
      "95617");
  BigInt<4> adjuster = *BigInt<4>::FromDecString(
      "944936681149208446651664254269745548490766851729442924617792859073125903"
      "783");
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // 104956732223578925557223488830078362736943595779422517109872604305846127009997
  input = *BigInt<4>::FromDecString(
      "358795011877924939067134028731607639090230656906793922100740187261350678"
      "7085");
  // 13836902468045855406793973610070160223721877582113063977376419473093071358947
  expected = *BigInt<4>::FromDecString(
      "211488592907427181113099752467384490255004996398674446817083716551085719"
      "82666");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // 58137146071470818113497440987533709949510360232295259246272891334885364755793
  input = *BigInt<4>::FromDecString(
      "567056632061005688596945785975957941913343351371449825599748549267295198"
      "3989");
  // 10012933272639613859587515636169426221587622189722693700891691228635545350648
  expected = *BigInt<4>::FromDecString(
      "448106334201548424302265130892044159754236703369274706140399820044324154"
      "3910");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // 28936367257625975338385437547494218524455841456842373037716593520344969509397
  input = *BigInt<4>::FromDecString(
      "394407580137911053895742321381425496701226931447443828555088855056098800"
      "3234");
  // 5533769493925008693547098095539268782323697506580128579981487108293743418881
  expected = *BigInt<4>::FromDecString(
      "137083209848048852584441963389817935909133205377178783422273787606625729"
      "27772");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a composite modulus and trivial representation
// of both input and output of the inversion method. The modulus
// is 2 plus the order of the scalar field of the bn254 curve
TEST(BYInverterTest, CompositeTrivial) {
  BigInt<4> modulus = *BigInt<4>::FromDecString(
      "218882428718392752222464057452572750885483644004160343436982041865758084"
      "95619");
  BigInt<4> adjuster = BigInt<4>::One();
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  input = *BigInt<4>::FromDecString(
      "260400902909203205801927137882405069909206412991261887835516351877431832"
      "0405");
  // Not invertible, since GCD(input, modulus) = 3
  EXPECT_FALSE(inverter.Invert(input, output));

  input = *BigInt<4>::FromDecString(
      "617886981998124808174262578967300334936713010129281877740500228549113711"
      "3070");
  expected = *BigInt<4>::FromDecString(
      "118091461884980844928468691940872607811557868933035705144433998283587672"
      "12457");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  input = *BigInt<4>::FromDecString(
      "118890046284150530256808306616576448503703129212685170076195830906438907"
      "09598");
  expected = *BigInt<4>::FromDecString(
      "954727260627150970144166832159552548250347273530028832700546847220916690"
      "8277");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a composite modulus and Montgomery representation of both
// input and output of the inversion method. The modulus is 2 plus the order
// of the scalar field of the bn254 curve. The Montgomery factor equals 2²⁵⁶.
// For the numbers specified in Montgomery representation their trivial form
// is in the comments
TEST(BYInverterTest, CompositeMontgomery) {
  BigInt<4> modulus = *BigInt<4>::FromDecString(
      "218882428718392752222464057452572750885483644004160343436982041865758084"
      "95619");
  BigInt<4> adjuster = *BigInt<4>::FromDecString(
      "157148295651647307182743343858683637233496413180705862961339672050283867"
      "5869");
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // 11865630156646177845488966194957438762902332631956652259274124517358851833759
  input = *BigInt<4>::FromDecString(
      "878201618454704726522650428350653751071479777045695423321764180703260530"
      "5262");
  // Not invertible, since GCD(input, modulus) = 3
  EXPECT_FALSE(inverter.Invert(input, output));

  // 13626187621506977415144012068495580987161044579599980007720002490790855281118
  input = *BigInt<4>::FromDecString(
      "112228015787800026601163349092648871925500177357276714834785886408856351"
      "41103");
  // 15569609600917656079795063009991863730429494071872866284038456731797547021054
  expected = *BigInt<4>::FromDecString(
      "183861344690836251394583500718777155185091441658607621272783642818974339"
      "70018");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // 4652503444426046901695366961113196520526566962552761000864601049682876484005
  input = *BigInt<4>::FromDecString(
      "591068310329360988454145556194259332147196340254859988568619407143112124"
      "7969");
  // 4735762809510279031273699449377321073262538802381459266345256675192986276306
  expected = *BigInt<4>::FromDecString(
      "962405928941411729669007908489132377986537009453884327835765191029297149"
      "4292");
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

}  // namespace tachyon::math
