# Commitments

The commitment scheme's interface has been designed to accommodate various commitment schemes.
While it might seem straightforward, it poses challenges, particularly in accessing the oracle.

For instance, in the case of a binary merkle tree, it commits leaves to a merkle root, constructing
the tree as a side effect. To facilitate this, we've implemented the `BinaryMerkleTree`, which holds
storage as a member. This design eliminates the need for an extra argument to obtain
the hash of the merkle tree during opening phase.

On the other hand, with the `KZG` commitment, there are no side effects during the commit phase.
However, during opening phase, it necessitates access to the original polynomials.
We're actively working on resolving this for future iterations.

This documentation provides an overview of each commitment scheme's interface in LaTeX,
offering a concise understanding of their functionalities.

## Vector Commitment Scheme

We've successfully unified `Commit()` to enable commitment production by passing a vector.
However, it's important to note that `Open()` and `Verify()` may have varying implementations
across schemes, leading to ambiguity. Please keep this in mind when working with them!

### Binary Merkle Tree

$$Commit(L) \to R$$
$$Open(i) \to \pi$$
$$Verify(R, \pi) \to true \space or \space false$$
$$L\text{: the set of leaf, }i\text{: index of the leaf }R\text{: merkle root}$$

## Univariate Polynomial Commitment Scheme

### FRI

$$Commit(P) \to C$$
$$Open(i) \to \pi$$
$$Verify(i, \pi) \to true \space or \space false$$
$$P\text{: polynomial, }i\text{: index of the leaf}$$

### SHPlonk

$$Commit(P) \to C$$
$$Open(O) \to \pi$$
$$Verify(O, \pi) \to true \space or \space false$$
$$P\text{: polynomial, }O\text{: polynomial openings }$$
