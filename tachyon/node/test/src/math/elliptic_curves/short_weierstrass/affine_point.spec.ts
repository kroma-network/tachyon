import { beforeAll, describe, expect, test } from '@jest/globals';

const tachyon = require('../../../../external/kroma_network_tachyon/tachyon/node/tachyon.node');

beforeAll(() => {
  tachyon.math.bn254.init();
});

describe('AffinePoint', () => {
  test('new AffinePoint()', () => {
    const p = new tachyon.math.bn254.G1AffinePoint();
    expect(p.x.isZero()).toBe(true);
    expect(p.y.isZero()).toBe(true);
    expect(p.infinity).toBe(true);

    const x = tachyon.math.bn254.Fq.random();
    const y = tachyon.math.bn254.Fq.random();
    const p2 = new tachyon.math.bn254.G1AffinePoint(x, y, false);
    expect(p2.x.eq(x)).toBe(true);
    expect(p2.y.eq(y)).toBe(true);
    expect(p2.infinity).toBe(false);
  });

  test('AffinePoint.zero()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.zero();
    expect(p.isZero()).toBe(true);
    expect(p.infinity).toBe(true);
  });

  test('AffinePoint.generator()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.isOnCurve()).toBe(true);
    expect(p.infinity).toBe(false);
  });

  test('AffinePoint.random()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.random();
    const p2 = tachyon.math.bn254.G1AffinePoint.random();
    expect(p.eq(p2)).toBe(false);
  });

  test('AffinePoint.eq()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.zero();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.eq(p)).toBe(true);
    expect(p2.eq(p2)).toBe(true);
    expect(p.eq(p2)).toBe(false);
    expect(p2.eq(p)).toBe(false);
  });

  test('AffinePoint.ne()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.zero();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.ne(p)).toBe(false);
    expect(p2.ne(p2)).toBe(false);
    expect(p.ne(p2)).toBe(true);
    expect(p2.ne(p)).toBe(true);
  });

  test('AffinePoint.add()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.add(p2).eq(new tachyon.math.bn254.G1JacobianPoint(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208560'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(4),
    ))).toBe(true);
  });

  test('AffinePoint.sub()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.sub(p2).isZero()).toBe(true);
  });

  test('AffinePoint.negative()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.negative().eq(new tachyon.math.bn254.G1AffinePoint(
      p.x,
      p.y.negative(),
    ))).toBe(true);
  });

  test('AffinePoint.double()', () => {
    const p = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.double().eq(new tachyon.math.bn254.G1JacobianPoint(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208560'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(4),
    ))).toBe(true);
  });
});
