# How to add elliptic curves

Follow this guide to add a new elliptic curve for Tachyon.

_Note_: We have our own development conventions. Please read the [conventions doc](/docs/how_to_contribute/conventions.md) before contributing.

## Add a `BUILD.bazel` file

Begin by creating a directory named `<new_elliptic_curve>` in `/tachyon/math/elliptic_curves/`. Add a `BUILD.bazel` file into this directory. Note that once parameters are added to `BUILD.bazel`, Bazel will automatically generate the elliptic curve code based on these parameters when it builds the target.

Next, implement the *scalar field* and *base field* for which the elliptic curve is defined on. See [here](/docs/how_to_contribute/math/how_to_add_prime_field.md) for how to add a *prime field*.

After implementing the *fields*, input the parameters of the elliptic curve into the generator(`generate_ec_points`) like shown below:

```bazel
load("//tachyon/math/elliptic_curves/short_weierstrass/generator:build_defs.bzl", "generate_ec_points")

generate_ec_points(
    name = "curve",
    # y² = x³ + ax + b
    a = ["{a}"],
    b = ["{b}"],
    base_field = "Fq",
    base_field_dep = ":fq",
    base_field_hdr = "tachyon/math/elliptic_curves/<new_elliptic_curve>/fq.h",
    gen_gpu = True, # False if the GPU code is not needed
    namespace = "tachyon::math::<new_elliptic_curve>",
    scalar_field = "Fr",
    scalar_field_dep = ":fr",
    scalar_field_hdr = "tachyon/math/elliptic_curves/<new_elliptic_curve>/fr.h",
    x = ["{x}"], # x-coordinate of the generator
    y = ["{y}"], # y-coordinate of the generator
)
```

_Note_: As of now, only elliptic curves in Short Weierstrass form are supported.

Once all these steps are completed, the new elliptic curve is now ready to be used in Tachyon!
