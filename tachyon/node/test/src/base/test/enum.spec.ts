import { describe, expect, test } from '@jest/globals';

const enum_test = require(
    '../../../external/kroma_network_tachyon/tachyon/node/base/test/test.node');

describe('enum', () => {
  test('enum values', () => {
    expect(enum_test.color.red).toBe(0);
    expect(enum_test.color.green).toBe(1);
    expect(enum_test.color.blue).toBe(2);
  });

  test('enum not writable', () => {
    expect(() => (enum_test.color.red = 1)).toThrow();
  });

  test('enum not configurable', () => {
    expect(() => delete enum_test.color.red).toThrow();
  });

  test('enum enumerable', () => {
    expect(enum_test.color.propertyIsEnumerable('red')).toBe(true);
  });
});
