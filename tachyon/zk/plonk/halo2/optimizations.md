# Optimizations

Contains a list of optimizations that have been made to the [scroll/halo2](https://github.com/scroll-tech/halo2) codebase. This is a living document and will be updated as new optimizations are made.

## 1. Lazy IFFT

Some parts of the Halo2 code hold polynomials in both coefficient and evaluation form. Considering that polynomials in coefficient form are only needed after squeezing y, we can delay IFFT until y is squeezed. This helps reduce memory pressure.

Relevant commit: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9)

### 1.1. Lazy IFFT for Permutation Argument

Relevant code optimization: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9#diff-a66badbb6b9b05ff643f192101f10cd59fe8edc0d9e63583f088859089c4e80fR202-R203).

Relevant halo2 code: [permutation product poly].

[permutation product poly]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/permutation/prover.rs#L170-L172

### 1.2. Lazy IFFT for Lookup Argument

Relevant code optimization: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9#diff-a66badbb6b9b05ff643f192101f10cd59fe8edc0d9e63583f088859089c4e80fR204-R205).

Relevant halo2 code: [permuted input poly], [permuted table poly] and [lookup product poly].

[permuted input poly]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/lookup/prover.rs#L136-L137
[permuted table poly]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/lookup/prover.rs#L139-L141
[lookup product poly]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/lookup/prover.rs#L291-L293

## 2. Batch Normalize

Originally, when committing to multiple polynomials sequentially, the conversion from xyzz point type to affine point is done individually for each commitment, which leads to a large number of **expensive** inverse operations. We optimize this inefficiency with **Batch Normalize**.

To set up **Batch Normalize**, conversions from xyzz point type to affine point are delayed until all commitments are calculated. Once all calculations are finished, xyzz point types are converted to affine points at once with [BatchNormalize()](https://github.com/kroma-network/tachyon/blob/522243b7b62b69c552dfa4768d9b7160f9e1694c/tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h#L113-L159). This reduces the number of required inverse operations to once per sequence.

Relevant commit: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9)

### 2.1. Batch Normalize for Permuted Polys

Relevant code optimization: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9#diff-a66badbb6b9b05ff643f192101f10cd59fe8edc0d9e63583f088859089c4e80fR150-R165).

Relevant halo2 code: [permuted input poly] and [permuted table poly].

### 2.2. Batch Normalize for Grand Product Polys

Relevant code optimization: [here](https://github.com/kroma-network/tachyon/pull/294/commits/0a9fe9c56c8d7bf9d193c89991003cee3d6a88b9#diff-a66badbb6b9b05ff643f192101f10cd59fe8edc0d9e63583f088859089c4e80fR172-R196).

Relevant halo2 code: [permutation product poly] and [lookup product poly].

### 2.3. Batch Inverse in Batch Normalize

Inverse operations are costly for elliptic curves. We can reduce the number of inverse operations on multiple points by using [BatchInverse()](https://github.com/kroma-network/tachyon/blob/48891d59d2d751665cb12bc65267d48450c847df/tachyon/math/base/groups.h#L152-L199), which only requires one inverse operation for all the points included.

We fully utilize **Batch Inverse** to avoid multiple inverse operations. For example, the original [batch_normalize()](https://github.com/zkcrypto/group/blob/0c5b04443b2b24c9a06d50ca890313b641f2a5df/src/lib.rs#L102-L110) inefficiently computed an inverse operation for every point, so we optimized our [BatchNormalize()](https://github.com/kroma-network/tachyon/blob/48891d59d2d751665cb12bc65267d48450c847df/tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h#L113-L159) to use the batch inverse operation to normalize all the points with a single inverse operation.

## 3. Optimizing parallelization code

We merge multiple parallelized parts into a single parallelized part for better performance by reducing the number of thread spawns and joins. Here are the list of examples:

| Description           | Tachyon       | Halo2         |
| --------------------- | ------------- | ------------- |
| Merge parallelization | [merging 1-t] | [merging 1-h] |
| Use collapse clause   | [merging 2-t] | [merging 2-h] |

[merging 1-t]: https://github.com/kroma-network/tachyon/pull/297/commits/d38013c2a335d75d7934b6c7539d2c9e1a95f504
[merging 1-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/evaluation.rs#L300-L715
[merging 2-t]: https://github.com/kroma-network/tachyon/commit/6d6273e2bf9f3f11583835f49099adbaba2b0954
[merging 2-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/poly/domain.rs#L181-L189

## 4. Optimizing MSM

Several optimizations have been made to the Multi-Scalar Multiplication (MSM) operation. For scalar multiplication, we have implemented [windowed non-adjacent form (WNAF)](https://github.com/kroma-network/tachyon/blob/655fdff6739c8b197c326aaac19c586c7b503dfd/tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger.h#L110-L160), which is a more efficient way of representing the scalar. For group operations, we have changed the bucket type to the PointXYZZ type, which is more efficient than the Jacobian type. View the benchmark result regarding the change to PointXYZZ type [here](https://github.com/kroma-network/tachyon/pull/36#issuecomment-1715653932).

## 5. Saving heap allocation

Considering the memory used by polynomials is typically high, reducing the number of memory allocations and reusing the existing allocated memory is very important.

| Description                                           | Tachyon      | Halo2        |
| ----------------------------------------------------- | ------------ | ------------ |
| Reuse compressed evals buffer                         | [saving 1-t] | [saving 1-h] |
| Reuse z buffer                                        | [saving 2-t] | [saving 2-h] |
| Compute l polys                                       | [saving 3-t] | [saving 3-h] |
| Avoid intermediate memory allocation when transposing | [saving 4-t] | [saving 4-h] |

[saving 1-t]: https://github.com/kroma-network/tachyon/commit/216288eac85c0699b8fccccfda01b3b17f47a8db
[saving 1-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/lookup/prover.rs#L105
[saving 2-t]: https://github.com/kroma-network/tachyon/commit/fa3a54c23e6afb62eb9bbff7841c87b9c2f6d7e6
[saving 2-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/permutation/prover.rs#L153
[saving 3-t]: https://github.com/kroma-network/tachyon/commit/f77a70d0c453db5a3c5c0bcfb69e0f13fbe1b599#diff-15682b00615ff3f3f158174b7a65e797e119d770ed7ee3b438fb0eb594a948e8
[saving 3-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/plonk/keygen.rs#L566-L592
[saving 4-t]: https://github.com/kroma-network/tachyon/pull/288/commits/44e089f5f0a83938e15b210af921c97d7d0868cf
[saving 4-h]: https://github.com/scroll-tech/halo2/blob/1070391642dd64b2d68b47ec246cba9e35bd3c15/halo2_proofs/src/poly/domain.rs#L182
