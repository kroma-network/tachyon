import { describe, expect, test } from '@jest/globals';

const functionTest = require('../../../external/kroma_network_tachyon/tachyon/node/base/test/function.node');

const wrongNumberOfArguments = TypeError('Wrong number of arguments');

function invalidArgument(i: number) {
  return TypeError(`Invalid argument #${i}`);
}

describe('function', () => {
  test('function std::string()', () => {
    expect(functionTest.hello()).toBe('world');
    expect(() => functionTest.hello(1)).toThrow(wrongNumberOfArguments);
  });

  test('function int(int, int) with default arguments', () => {
    expect(functionTest.sum()).toBe(3);
    expect(functionTest.sum(2)).toBe(4);
    expect(functionTest.sum(2, 3)).toBe(5);
    expect(() => functionTest.sum(1, 'abc')).toThrow(invalidArgument(1));
    expect(() => functionTest.sum(2, 3, 4)).toThrow(wrongNumberOfArguments);
  });

  test('function void()', () => {
    expect(() => functionTest.do_nothing()).not.toThrow();
    expect(() => functionTest.do_nothing(1)).toThrow(wrongNumberOfArguments);
  });
});
