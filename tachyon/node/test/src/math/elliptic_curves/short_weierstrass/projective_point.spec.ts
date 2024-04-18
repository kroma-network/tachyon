import { beforeAll, describe, expect, test } from '@jest/globals';

const tachyon = require('../../../../external/kroma_network_tachyon/tachyon/node/tachyon.node');

beforeAll(() => {
  tachyon.math.bn254.init();
});

describe('ProjectivePoint', () => {
  test('new ProjectivePoint()', () => {
    const p = new tachyon.math.bn254.G1ProjectivePoint();
    expect(p.x.isOne()).toBe(true);
    expect(p.y.isOne()).toBe(true);
    expect(p.z.isZero()).toBe(true);

    const x = tachyon.math.bn254.Fq.random();
    const y = tachyon.math.bn254.Fq.random();
    const z = tachyon.math.bn254.Fq.random();
    const p2 = new tachyon.math.bn254.G1ProjectivePoint(x, y, z);
    expect(p2.x.eq(x)).toBe(true);
    expect(p2.y.eq(y)).toBe(true);
    expect(p2.z.eq(z)).toBe(true);
  });

  test('ProjectivePoint.zero()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.zero();
    expect(p.isZero()).toBe(true);
    expect(p.z.isZero()).toBe(true);
  });

  test('ProjectivePoint.generator()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.isOnCurve()).toBe(true);
    expect(p.z.isZero()).toBe(false);
  });

  test('ProjectivePoint.random()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.random();
    const p2 = tachyon.math.bn254.G1ProjectivePoint.random();
    expect(p.eq(p2)).toBe(false);
  });

  test('ProjectivePoint.eq()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.zero();
    const p2 = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.eq(p)).toBe(true);
    expect(p2.eq(p2)).toBe(true);
    expect(p.eq(p2)).toBe(false);
    expect(p2.eq(p)).toBe(false);
  });

  test('ProjectivePoint.ne()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.zero();
    const p2 = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.ne(p)).toBe(false);
    expect(p2.ne(p2)).toBe(false);
    expect(p.ne(p2)).toBe(true);
    expect(p2.ne(p)).toBe(true);
  });

  test('ProjectivePoint.add()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    const p2 = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.add(p2).eq(new tachyon.math.bn254.G1ProjectivePoint(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208491'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });

  test('ProjectivePoint.addMixed()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.addMixed(p2).eq(new tachyon.math.bn254.G1ProjectivePoint(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208491'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });

  test('ProjectivePoint.sub()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    const p2 = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.sub(p2).isZero()).toBe(true);
  });

  test('ProjectivePoint.subMixed()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    const p2 = tachyon.math.bn254.G1AffinePoint.generator();
    expect(p.subMixed(p2).isZero()).toBe(true);
  });

  test('ProjectivePoint.negate()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.negate().eq(new tachyon.math.bn254.G1ProjectivePoint(
      p.x,
      p.y.negate(),
      p.z,
    ))).toBe(true);
  });

  test('ProjectivePoint.double()', () => {
    const p = tachyon.math.bn254.G1ProjectivePoint.generator();
    expect(p.double().eq(new tachyon.math.bn254.G1ProjectivePoint(
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208491'),
      tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208572'),
      tachyon.math.bn254.Fq.fromNumber(64),
    ))).toBe(true);
  });
});
