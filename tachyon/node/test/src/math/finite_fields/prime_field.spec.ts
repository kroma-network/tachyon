import { beforeAll, describe, expect, test } from '@jest/globals';

const tachyon = require('../../../external/kroma_network_tachyon/tachyon/node/tachyon.node');

beforeAll(() => {
  tachyon.math.bn254.init();
});

describe('PrimeField', () => {
  test('PrimeField.zero()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    expect(f.isZero()).toBe(true);
    expect(f.isOne()).toBe(false);
  });

  test('PrimeField.one()', () => {
    const f = tachyon.math.bn254.Fq.one();
    expect(f.isZero()).toBe(false);
    expect(f.isOne()).toBe(true);
  });

  test('PrimeField.random()', () => {
    const f = tachyon.math.bn254.Fq.random();
    const f2 = tachyon.math.bn254.Fq.random();
    expect(f.eq(f2)).toBe(false);
  });

  test('PrimeField.fromDecString()', () => {
    const f = tachyon.math.bn254.Fq.fromDecString('12345678910');
    expect(f.toString()).toBe('12345678910');
    expect(f.toHexString()).toBe('0x2dfdc1c3e');
  });

  test('PrimeField.fromHexString()', () => {
    const f = tachyon.math.bn254.Fq.fromHexString('0x2dfdc1c3e');
    expect(f.toString()).toBe('12345678910');
    expect(f.toHexString()).toBe('0x2dfdc1c3e');
  });

  test('PrimeField.eq()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.eq(f)).toBe(true);
    expect(f2.eq(f2)).toBe(true);
    expect(f.eq(f2)).toBe(false);
    expect(f2.eq(f)).toBe(false);
  });

  test('PrimeField.ne()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.ne(f)).toBe(false);
    expect(f2.ne(f2)).toBe(false);
    expect(f.ne(f2)).toBe(true);
    expect(f2.ne(f)).toBe(true);
  });

  test('PrimeField.lt()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.lt(f)).toBe(false);
    expect(f2.lt(f2)).toBe(false);
    expect(f.lt(f2)).toBe(true);
    expect(f2.lt(f)).toBe(false);
  });

  test('PrimeField.le()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.le(f)).toBe(true);
    expect(f2.le(f2)).toBe(true);
    expect(f.le(f2)).toBe(true);
    expect(f2.le(f)).toBe(false);
  });

  test('PrimeField.gt()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.gt(f)).toBe(false);
    expect(f2.gt(f2)).toBe(false);
    expect(f.gt(f2)).toBe(false);
    expect(f2.gt(f)).toBe(true);
  });

  test('PrimeField.ge()', () => {
    const f = tachyon.math.bn254.Fq.zero();
    const f2 = tachyon.math.bn254.Fq.one();
    expect(f.ge(f)).toBe(true);
    expect(f2.ge(f2)).toBe(true);
    expect(f.ge(f2)).toBe(false);
    expect(f2.ge(f)).toBe(true);
  });

  test('PrimeField.add()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    const f2 = tachyon.math.bn254.Fq.fromNumber(3);
    expect(f.add(f2).eq(tachyon.math.bn254.Fq.fromNumber(7))).toBe(true);
  });

  test('PrimeField.sub()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    const f2 = tachyon.math.bn254.Fq.fromNumber(3);
    expect(f.sub(f2).eq(tachyon.math.bn254.Fq.fromNumber(1))).toBe(true);
  });

  test('PrimeField.mul()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    const f2 = tachyon.math.bn254.Fq.fromNumber(3);
    expect(f.mul(f2).eq(tachyon.math.bn254.Fq.fromNumber(12))).toBe(true);
  });

  test('PrimeField.div()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    const f2 = tachyon.math.bn254.Fq.fromNumber(3);
    expect(f.div(f2).eq(tachyon.math.bn254.Fq.fromDecString('14592161914559516814830937163504850059130874104865215775126025263096817472390'))).toBe(true);
  });

  test('PrimeField.negative()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    expect(f.negative().eq(tachyon.math.bn254.Fq.fromDecString('21888242871839275222246405745257275088696311157297823662689037894645226208579'))).toBe(true);
  });

  test('PrimeField.double()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    expect(f.double().eq(tachyon.math.bn254.Fq.fromNumber(8))).toBe(true);
  });

  test('PrimeField.square()', () => {
    const f = tachyon.math.bn254.Fq.fromNumber(4);
    expect(f.square().eq(tachyon.math.bn254.Fq.fromNumber(16))).toBe(true);
  });
});
