from absl.testing import absltest

from external.kroma_network_tachyon.tachyon.py import tachyon


class PointXYZZTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        tachyon.math.bn254.init()

    def test_constructor(self):
        p = tachyon.math.bn254.G1PointXYZZ()
        self.assertTrue(p.x.is_one())
        self.assertTrue(p.y.is_one())
        self.assertTrue(p.zz.is_zero())
        self.assertTrue(p.zzz.is_zero())

        x = tachyon.math.bn254.Fq.random()
        y = tachyon.math.bn254.Fq.random()
        zz = tachyon.math.bn254.Fq.random()
        zzz = tachyon.math.bn254.Fq.random()
        p2 = tachyon.math.bn254.G1PointXYZZ(x, y, zz, zzz)
        self.assertEqual(p2.x, x)
        self.assertEqual(p2.y, y)
        self.assertEqual(p2.zz, zz)
        self.assertEqual(p2.zzz, zzz)

    def test_zero(self):
        p = tachyon.math.bn254.G1PointXYZZ.zero()
        self.assertTrue(p.is_zero())
        self.assertTrue(p.zz.is_zero())

    def test_generator(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        self.assertTrue(p.is_on_curve())
        self.assertFalse(p.zz.is_zero())

    def test_random(self):
        p = tachyon.math.bn254.G1PointXYZZ.random()
        p2 = tachyon.math.bn254.G1PointXYZZ.random()
        self.assertNotEqual(p, p2)

    def test_eq(self):
        p = tachyon.math.bn254.G1PointXYZZ.zero()
        p2 = tachyon.math.bn254.G1PointXYZZ.generator()
        self.assertTrue(p == p)
        self.assertTrue(p2 == p2)
        self.assertFalse(p == p2)
        self.assertFalse(p2 == p)

    def test_ne(self):
        p = tachyon.math.bn254.G1PointXYZZ.zero()
        p2 = tachyon.math.bn254.G1PointXYZZ.generator()
        self.assertFalse(p != p)
        self.assertFalse(p2 != p2)
        self.assertTrue(p != p2)
        self.assertTrue(p2 != p)

    def test_add(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        p2 = tachyon.math.bn254.G1PointXYZZ.generator()
        expected = tachyon.math.bn254.G1PointXYZZ(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(16),
            tachyon.math.bn254.Fq(64)
        )
        self.assertEqual(p + p2, expected)
        p += p2
        self.assertEqual(p, expected)

        p = tachyon.math.bn254.G1PointXYZZ.generator()
        ap2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertEqual(p + ap2, expected)
        p += ap2
        self.assertEqual(p, expected)

    def test_sub(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        p2 = tachyon.math.bn254.G1PointXYZZ.generator()
        self.assertTrue((p - p2).is_zero())
        p -= p2
        self.assertTrue(p.is_zero())

        p = tachyon.math.bn254.G1PointXYZZ.generator()
        ap2 = tachyon.math.bn254.G1AffinePoint.generator()
        self.assertTrue((p - ap2).is_zero())
        p -= ap2
        self.assertTrue((p).is_zero())

    def test_scalar_mul(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        n = tachyon.math.bn254.Fr(2)
        expected = tachyon.math.bn254.G1PointXYZZ(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(16),
            tachyon.math.bn254.Fq(64)
        )
        self.assertEqual(p * n, expected)
        self.assertEqual(n * p, expected)
        p *= n
        self.assertEqual(p, expected)

    def test_negative(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        self.assertEqual(-p, tachyon.math.bn254.G1PointXYZZ(
            p.x,
            -p.y,
            p.zz,
            p.zzz
        ))

    def test_double(self):
        p = tachyon.math.bn254.G1PointXYZZ.generator()
        expected = tachyon.math.bn254.G1PointXYZZ(
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208560'),
            tachyon.math.bn254.Fq.from_dec_string(
                '21888242871839275222246405745257275088696311157297823662689037894645226208572'),
            tachyon.math.bn254.Fq(16),
            tachyon.math.bn254.Fq(64)
        )
        self.assertEqual(p.double(), expected)
        p.double_in_place()
        self.assertEqual(p, expected)
