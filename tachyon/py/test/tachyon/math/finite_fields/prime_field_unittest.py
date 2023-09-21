from absl.testing import absltest

from external.kroma_network_tachyon.tachyon.py import tachyon


class PrimeFieldTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        tachyon.math.bn254.init()

    def test_zero(self):
        f = tachyon.math.bn254.Fq.zero()
        self.assertTrue(f.is_zero())
        self.assertFalse(f.is_one())

    def test_one(self):
        f = tachyon.math.bn254.Fq.one()
        self.assertFalse(f.is_zero())
        self.assertTrue(f.is_one())

    def test_random(self):
        f = tachyon.math.bn254.Fq.random()
        f2 = tachyon.math.bn254.Fq.random()
        self.assertNotEqual(f, f2)

    def test_from_dec_string(self):
        f = tachyon.math.bn254.Fq.from_dec_string('12345678910')
        self.assertEqual(f.to_string(), '12345678910')
        self.assertEqual(f.to_hex_string(), '0x2dfdc1c3e')

    def test_from_hex_string(self):
        f = tachyon.math.bn254.Fq.from_hex_string('0x2dfdc1c3e')
        self.assertEqual(f.to_string(), '12345678910')
        self.assertEqual(f.to_hex_string(), '0x2dfdc1c3e')

    def test_eq(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertTrue(f == f)
        self.assertTrue(f2 == f2)
        self.assertFalse(f == f2)
        self.assertFalse(f2 == f)

    def test_ne(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertFalse(f != f)
        self.assertFalse(f2 != f2)
        self.assertTrue(f != f2)
        self.assertTrue(f2 != f)

    def test_lt(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertFalse(f < f)
        self.assertFalse(f2 < f2)
        self.assertTrue(f < f2)
        self.assertFalse(f2 < f)

    def test_le(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertTrue(f <= f)
        self.assertTrue(f2 <= f2)
        self.assertTrue(f <= f2)
        self.assertFalse(f2 <= f)

    def test_gt(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertFalse(f > f)
        self.assertFalse(f2 > f2)
        self.assertFalse(f > f2)
        self.assertTrue(f2 > f)

    def test_ge(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertTrue(f >= f)
        self.assertTrue(f2 >= f2)
        self.assertFalse(f >= f2)
        self.assertTrue(f2 >= f)

    def test_ge(self):
        f = tachyon.math.bn254.Fq.zero()
        f2 = tachyon.math.bn254.Fq.one()
        self.assertTrue(f >= f)
        self.assertTrue(f2 >= f2)
        self.assertFalse(f >= f2)
        self.assertTrue(f2 >= f)

    def test_add(self):
        p = tachyon.math.bn254.Fq(4)
        p2 = tachyon.math.bn254.Fq(3)
        expected = tachyon.math.bn254.Fq(7)
        self.assertEqual(p + p2, expected)
        p += p2
        self.assertEqual(p, expected)

    def test_sub(self):
        p = tachyon.math.bn254.Fq(4)
        p2 = tachyon.math.bn254.Fq(3)
        expected = tachyon.math.bn254.Fq(1)
        self.assertEqual(p - p2, expected)
        p -= p2
        self.assertEqual(p, expected)

    def test_mul(self):
        p = tachyon.math.bn254.Fq(4)
        p2 = tachyon.math.bn254.Fq(3)
        expected = tachyon.math.bn254.Fq(12)
        self.assertEqual(p * p2, expected)
        p *= p2
        self.assertEqual(p, expected)

    def test_div(self):
        p = tachyon.math.bn254.Fq(4)
        p2 = tachyon.math.bn254.Fq(3)
        expected = tachyon.math.bn254.Fq.from_dec_string(
            '14592161914559516814830937163504850059130874104865215775126025263096817472390')
        self.assertEqual(p / p2, expected)
        p /= p2
        self.assertEqual(p, expected)

    def test_negative(self):
        f = tachyon.math.bn254.Fq(4)
        self.assertEqual(-f, tachyon.math.bn254.Fq.from_dec_string(
            '21888242871839275222246405745257275088696311157297823662689037894645226208579'))

    def test_double(self):
        f = tachyon.math.bn254.Fq(4)
        expected = tachyon.math.bn254.Fq(8)
        self.assertEqual(f.double(), expected)
        f.double_in_place()
        self.assertEqual(f, expected)

    def test_square(self):
        f = tachyon.math.bn254.Fq(4)
        expected = tachyon.math.bn254.Fq(16)
        self.assertEqual(f.square(), expected)
        f.square_in_place()
        self.assertEqual(f, expected)
