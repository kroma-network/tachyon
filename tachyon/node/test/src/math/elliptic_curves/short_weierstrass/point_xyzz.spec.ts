import { beforeAll, describe, expect, test } from '@jest/globals';

const tachyon = require('../../../../external/kroma_network_tachyon/tachyon/node/tachyon.node');

beforeAll(() => {
  tachyon.math.bn254.init();
});

describe('PointXYZZ', () => {
  test('new PointXYZZ()', () => {
    const p = new tachyon.math.bn254.G1PointXYZZ();
    expect(p.x.isOne()).toBe(true);
    expect(p.y.isOne()).toBe(true);
    expect(p.zz.isZero()).toBe(true);
    expect(p.zzz.isZero()).toBe(true);

    const x = tachyon.math.bn254.Fq.random();
    const y = tachyon.math.bn254.Fq.random();
    const zz = tachyon.math.bn254.Fq.random();
    const zzz = tachyon.math.bn254.Fq.random();
    const p2 = new tachyon.math.bn254.G1PointXYZZ(x, y, zz, zzz);
    expect(p2.x.eq(x)).toBe(true);
    expect(p2.y.eq(y)).toBe(true);
    expect(p2.zz.eq(zz)).toBe(true);
    expect(p2.zzz.eq(zzz)).toBe(true);
  });

  test('PointXYZZ.zero()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.zero();
    expect(p.isZero()).toBe(true);
    expect(p.zz.isZero()).toBe(true);
  });

  test('PointXYZZ.generator()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.isOnCurve()).toBe(true);
    expect(p.zz.isZero()).toBe(false);
  });

  test('PointXYZZ.random()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.random();
    const p2 = tachyon.math.bn254.G1PointXYZZ.random();
    expect(p.eq(p2)).toBe(false);
  });

  test('PointXYZZ.eq()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.zero();
    const p2 = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.eq(p)).toBe(true);
    expect(p2.eq(p2)).toBe(true);
    expect(p.eq(p2)).toBe(false);
    expect(p2.eq(p)).toBe(false);
  });

  test('PointXYZZ.ne()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.zero();
    const p2 = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.ne(p)).toBe(false);
    expect(p2.ne(p2)).toBe(false);
    expect(p.ne(p2)).toBe(true);
    expect(p2.ne(p)).toBe(true);
  });

  test('PointXYZZ.add()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    const p2 = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.add(p2).eq(new tachyon.math.bn254.G1PointXYZZ(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208560'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(16),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });

  test('PointXYZZ.addMixed()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.addMixed(p2).eq(new tachyon.math.bn254.G1PointXYZZ(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208560'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(16),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });

  test('PointXYZZ.sub()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    const p2 = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.sub(p2).isZero()).toBe(true);
  });

  test('PointXYZZ.subMixed()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.subMixed(p2).isZero()).toBe(true);
  });

  test('PointXYZZ.negative()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.negative().eq(new tachyon.math.bn254.G1PointXYZZ(
      p.x,
      p.y.negative(),
      p.zz,
      p.zzz,
    ))).toBe(true);
  });

  test('PointXYZZ.double()', () => {
    const p = tachyon.math.bn254.G1PointXYZZ.generator();
    expect(p.double().eq(new tachyon.math.bn254.G1PointXYZZ(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208560'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(16),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });
});
