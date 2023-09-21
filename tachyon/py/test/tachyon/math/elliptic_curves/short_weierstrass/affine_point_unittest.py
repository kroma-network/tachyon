from absl.testing import absltest

from external.kroma_network_tachyon.tachyon.py import tachyon


class AffinePointTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        tachyon.math.bn254.init()

    def test_constructor(self):
        p = tachyon.math.bn254.G1AffinePoint()
        self.assertTrue(p.x.is_zero())
        self.assertTrue(p.y.is_zero())
        self.assertTrue(p.infinity)

        x = tachyon.math.bn254.Fq.random()
        y = tachyon.math.bn254.Fq.random()
        p2 = tachyon.math.bn254.G1AffinePoint(x, y, False)
        self.assertEqual(p2.x, x)
        self.assertEqual(p2.y, y)
        self.assertEqual(p2.infinity, False)

    def test_zero(self):
        p = tachyon.math.bn254.G1AffinePoint.zero()
        self.assertTrue(p.is_zero())
        self.assertTrue(p.infinity)

    def test_generator(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertTrue(p.is_on_curve())
        self.assertFalse(p.infinity)

    def test_random(self):
        p = tachyon.math.bn254.G1AffinePoint.random()
        p2 = tachyon.math.bn254.G1AffinePoint.random()
        self.assertNotEqual(p, p2)

    def test_eq(self):
        p = tachyon.math.bn254.G1AffinePoint.zero()
        p2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertTrue(p == p)
        self.assertTrue(p2 == p2)
        self.assertFalse(p == p2)
        self.assertFalse(p2 == p)

    def test_ne(self):
        p = tachyon.math.bn254.G1AffinePoint.zero()
        p2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertFalse(p != p)
        self.assertFalse(p2 != p2)
        self.assertTrue(p != p2)
        self.assertTrue(p2 != p)

    def test_add(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        p2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertEqual(p + p2, tachyon.math.bn254.G1JacobianPoint(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(4)
        ))

    def test_sub(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        p2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertTrue((p - p2).is_zero())

    def test_scalar_mul(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        n = tachyon.math.bn254.Fr(2)
        expected = tachyon.math.bn254.G1JacobianPoint(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(4))
        self.assertEqual(p * n, expected)
        self.assertEqual(n * p, expected)

    def test_negative(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertEqual(-p, tachyon.math.bn254.G1AffinePoint(
            p.x,
            -p.y
        ))

    def test_double(self):
        p = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertEqual(p.double(), tachyon.math.bn254.G1JacobianPoint(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(4)
        ))
