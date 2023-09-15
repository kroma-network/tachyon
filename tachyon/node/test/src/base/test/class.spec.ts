import { describe, expect, test } from '@jest/globals';

const class_test = require('../../../external/kroma_network_tachyon/tachyon/node/base/test/class.node');

const callNonConstMethod = TypeError('Call non-const method');
const noSuchConstructor = TypeError('No such constructor');

describe('class', () => {
  test('method setter/getter', () => {
    let p: any = null;
    expect(() => (p = new class_test.Point())).not.toThrow();
    expect(() => p.setX(1)).not.toThrow();
    expect(p.getX()).toBe(1);
    expect(() => p.setY(2)).not.toThrow();
    expect(p.getY()).toBe(2);
  });

  test('property setter/getter', () => {
    let p: any = null;
    expect(() => (p = new class_test.Point())).not.toThrow();
    expect(() => (p.propertyX = 1)).not.toThrow();
    expect(p.propertyX).toBe(1);
    expect(() => (p.propertyY = 2)).not.toThrow();
    expect(p.propertyY).toBe(2);
    expect(p.getX()).toBe(1);
    expect(p.getY()).toBe(2);
  });

  test('member setter/getter', () => {
    let p: any = null;
    expect(() => (p = new class_test.Point())).not.toThrow();
    expect(() => (p.x = 1)).not.toThrow();
    expect(p.x).toBe(1);
    expect(() => (p.y = 2)).not.toThrow();
    expect(p.y).toBe(2);
    expect(p.getX()).toBe(1);
    expect(p.getY()).toBe(2);
  });

  test('static method setter/getter', () => {
    expect(() => class_test.Point.setDimension(3)).not.toThrow();
    expect(class_test.Point.getDimension()).toBe(3);
  });

  test('static property setter/getter', () => {
    expect(() => (class_test.Point.propertyDimension = 3)).not.toThrow();
    expect(class_test.Point.propertyDimension).toBe(3);
    expect(class_test.Point.getDimension()).toBe(3);
  });

  test('static member setter/getter', () => {
    expect(() => (class_test.Point.dimension = 3)).not.toThrow();
    expect(class_test.Point.dimension).toBe(3);
    expect(class_test.Point.getDimension()).toBe(3);
  });

  test('default arguments', () => {
    let adder: any = null;
    expect(() => (adder = new class_test.Adder())).not.toThrow();
    expect(adder.add()).toBe(10);
    expect(adder.add(2)).toBe(21);
    expect(adder.add(2, 3)).toBe(33);
    expect(adder.add(2, 3, 4)).toBe(46);
    expect(adder.add(2, 3, 4, 5)).toBe(60);

    expect(class_test.Adder.sAdd()).toBe(10);
    expect(class_test.Adder.sAdd(2)).toBe(11);
    expect(class_test.Adder.sAdd(2, 3)).toBe(12);
    expect(class_test.Adder.sAdd(2, 3, 4)).toBe(13);
    expect(class_test.Adder.sAdd(2, 3, 4, 5)).toBe(14);
  });

  test('reference method setter/getter', () => {
    let rect: any = null;
    expect(() => (rect = new class_test.Rect())).not.toThrow();
    expect(() => (rect.getTopLeft().x = 1)).not.toThrow();
    expect(rect.getTopLeft().x).toBe(1);
    expect(() => (rect.getTopLeft().y = 2)).not.toThrow();
    expect(rect.getTopLeft().y).toBe(2);
  });

  test('reference member setter/getter', () => {
    let rect: any = null;
    expect(() => (rect = new class_test.Rect())).not.toThrow();
    expect(() => (rect.topLeft.x = 1)).not.toThrow();
    expect(rect.topLeft.x).toBe(1);
    expect(() => (rect.topLeft.y = 2)).not.toThrow();
    expect(rect.topLeft.y).toBe(2);
    expect(rect.getTopLeft().x).toBe(1);
    expect(rect.getTopLeft().y).toBe(2);
  });

  test('const reference method setter/getter', () => {
    let rect: any = null;
    expect(() => (rect = new class_test.Rect())).not.toThrow();
    expect(() => rect.getConstTopLeft().setX(1)).toThrow(callNonConstMethod);
    expect(() => rect.getConstTopLeft().setY(2)).toThrow(callNonConstMethod);
    expect(() => (rect.getConstTopLeft().x = 1)).toThrow(callNonConstMethod);
    expect(() => (rect.getConstTopLeft().y = 2)).toThrow(callNonConstMethod);
  });

  test('pass const reference as an argument', () => {
    expect(
      class_test.Point.distance(
        new class_test.Point(),
        new class_test.Point(3, 4),
      ),
    ).toBe(5);
  });

  test('pass reference as an argument', () => {
    let p: any;
    expect(() => {
      p = new class_test.Point(3, 4);
      class_test.doubleWithReference(p);
    }).not.toThrow();
    expect(p.x).toBe(6);
    expect(p.y).toBe(8);
    expect(() =>
      class_test.doubleWithReference(new class_test.Rect()),
    ).toThrow();
    // TODO(chokobole): Enable these tests
    // expect(() => class_test.doubleWithSharedPtr(p)).toThrow();
    // expect(() => class_test.doubleWithUniquePtr(p)).toThrow();
  });

  test('pass value as an argument', () => {
    let p: any,
      q: any = null;
    expect(() => {
      p = new class_test.Point(3, 4);
      q = class_test.doubleWithValue(p);
    }).not.toThrow();
    expect(p.x).toBe(3);
    expect(p.y).toBe(4);
    expect(q.x).toBe(6);
    expect(q.y).toBe(8);
    expect(() => class_test.doubleWithValue(new class_test.Rect())).toThrow();
    // TODO(chokobole): Enable these tests
    // expect(() => class_test.doubleWithSharedPtr(p)).toThrow();
    // expect(() => class_test.doubleWithUniquePtr(p)).toThrow();
  });

  test('construct with object', () => {
    let r: any;
    expect(() => {
      r = new class_test.Rect(
        new class_test.Point(1, 2),
        new class_test.Point(3, 4),
      );
    }).not.toThrow();
    expect(r.topLeft.x).toBe(1);
    expect(r.topLeft.y).toBe(2);
    expect(r.bottomRight.x).toBe(3);
    expect(r.bottomRight.y).toBe(4);
  });

  test('constructor overloadding', () => {
    let v: any;
    expect(() => (v = new class_test.Variant(true))).not.toThrow();
    expect(v.b).toBe(true);
    expect(() => (v = new class_test.Variant(5))).not.toThrow();
    expect(v.i).toBe(5);
    expect(() => (v = new class_test.Variant(BigInt(10)))).not.toThrow();
    expect(v.i64).toBe(BigInt(10));
    expect(() => (v = new class_test.Variant('hello'))).not.toThrow();
    expect(v.s).toBe('hello');
    expect(
      () => (v = new class_test.Variant(new Array(1, 2, 3))),
    ).not.toThrow();
    expect(v.ivec).toStrictEqual(new Array(1, 2, 3));
    expect(() => (v = new class_test.Variant(1, 'hello'))).not.toThrow();
    expect(v.i).toBe(1);
    expect(v.s).toBe('hello');
    expect(
      () => (v = new class_test.Variant(new class_test.Point(1, 2))),
    ).not.toThrow();
    expect(v.p.x).toBe(1);
    expect(v.p.y).toBe(2);
    expect(() => (v = new class_test.Variant(new class_test.Rect()))).toThrow(
      noSuchConstructor,
    );
  });

  test('inheritance', () => {
    let cp: any;
    expect(
      () => (cp = new class_test.ColoredPoint(1, 2, class_test.color.red)),
    ).not.toThrow();
    expect(() => class_test.doubleWithValue(cp)).not.toThrow();
    expect(cp.x).toBe(1);
    expect(cp.y).toBe(2);
    expect(() => class_test.doubleWithReference(cp)).not.toThrow();
    expect(cp.x).toBe(2);
    expect(cp.y).toBe(4);
  });

  // TODO(chokobole): Enable these tests
  // test('shared_ptr', () => {
  //   let p: any;
  //   expect(() => (p = new class_test.Point(1, 2))).not.toThrow();
  //   expect(() => class_test.doubleWithReference(p)).not.toThrow();
  //   expect(p.x).toBe(2);
  //   expect(p.y).toBe(4);
  //   expect(() => class_test.doubleWithSharedPtr(p)).not.toThrow();
  //   expect(p.x).toBe(4);
  //   expect(p.y).toBe(8);
  //   expect(() => class_test.doubleWithUniquePtr(p)).toThrow();
  // });

  // test('unique_ptr', () => {
  //   let p: any;
  //   expect(() => (p = new class_test.Point(1, 2))).not.toThrow();
  //   expect(() => class_test.doubleWithReference(p)).not.toThrow();
  //   expect(p.x).toBe(2);
  //   expect(p.y).toBe(4);
  //   expect(() => class_test.doubleWithSharedPtr(p)).toThrow();
  //   expect(() => class_test.doubleWithUniquePtr(p)).not.toThrow();
  // });
});
