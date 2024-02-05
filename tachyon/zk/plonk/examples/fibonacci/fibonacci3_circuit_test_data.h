#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI3_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI3_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

namespace tachyon::zk::plonk::halo2 {

namespace fibonacci3 {

constexpr char kPinnedConstraintSystem[] =
    "PinnedConstraintSystem { num_fixed_columns: 0, num_advice_columns: 5, "
    "num_instance_columns: 0, num_selectors: 1, gates: "
    "[Product(Product(Selector(Selector(0, true)), Sum(Advice { query_index: "
    "0, column_index: 0, rotation: Rotation(0) }, Negated(Advice { "
    "query_index: 1, column_index: 1, rotation: Rotation(0) }))), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))), Product(Selector(Selector(0, true)), "
    "Product(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) }))), Sum(Advice { query_index: 4, column_index: 3, "
    "rotation: Rotation(0) }, Negated(Advice { query_index: 3, column_index: "
    "2, rotation: Rotation(0) })))), Product(Product(Selector(Selector(0, "
    "true)), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))))), Sum(Advice { query_index: 4, column_index: "
    "3, rotation: Rotation(0) }, Negated(Sum(Advice { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, Negated(Advice { query_index: "
    "1, column_index: 1, rotation: Rotation(0) })))))], advice_queries: "
    "[(Column { index: 0, column_type: Advice }, Rotation(0)), (Column { "
    "index: 1, column_type: Advice }, Rotation(0)), (Column { index: 4, "
    "column_type: Advice }, Rotation(0)), (Column { index: 2, column_type: "
    "Advice }, Rotation(0)), (Column { index: 3, column_type: Advice }, "
    "Rotation(0))], instance_queries: [], fixed_queries: [], permutation: "
    "Argument { columns: [] }, lookups: [], constants: [], minimum_degree: "
    "None }";

constexpr uint8_t kExpectedProof[] = {
    33,  231, 248, 121, 132, 226, 31,  179, 86,  67,  213, 109, 201, 53,  147,
    93,  41,  30,  16,  172, 115, 107, 0,   172, 225, 38,  5,   143, 136, 68,
    3,   47,  222, 110, 120, 246, 167, 202, 30,  64,  192, 113, 187, 20,  209,
    5,   19,  35,  161, 55,  99,  167, 153, 242, 153, 84,  169, 135, 224, 148,
    140, 22,  18,  148, 5,   145, 196, 201, 20,  216, 6,   183, 82,  73,  62,
    87,  36,  44,  76,  29,  120, 103, 124, 253, 87,  100, 218, 122, 245, 84,
    26,  140, 240, 97,  130, 134, 92,  104, 200, 22,  60,  26,  130, 227, 203,
    220, 139, 80,  219, 111, 129, 53,  194, 0,   185, 59,  50,  147, 22,  2,
    25,  58,  255, 70,  194, 103, 39,  5,   143, 199, 14,  63,  41,  37,  111,
    200, 181, 194, 119, 182, 175, 185, 182, 254, 90,  205, 250, 56,  35,  47,
    214, 38,  249, 166, 235, 34,  66,  177, 210, 132, 33,  231, 248, 121, 132,
    226, 31,  179, 86,  67,  213, 109, 201, 53,  147, 93,  41,  30,  16,  172,
    115, 107, 0,   172, 225, 38,  5,   143, 136, 68,  3,   47,  222, 110, 120,
    246, 167, 202, 30,  64,  192, 113, 187, 20,  209, 5,   19,  35,  161, 55,
    99,  167, 153, 242, 153, 84,  169, 135, 224, 148, 140, 22,  18,  148, 5,
    145, 196, 201, 20,  216, 6,   183, 82,  73,  62,  87,  36,  44,  76,  29,
    120, 103, 124, 253, 87,  100, 218, 122, 245, 84,  26,  140, 240, 97,  130,
    134, 92,  104, 200, 22,  60,  26,  130, 227, 203, 220, 139, 80,  219, 111,
    129, 53,  194, 0,   185, 59,  50,  147, 22,  2,   25,  58,  255, 70,  194,
    103, 39,  5,   143, 199, 14,  63,  41,  37,  111, 200, 181, 194, 119, 182,
    175, 185, 182, 254, 90,  205, 250, 56,  35,  47,  214, 38,  249, 166, 235,
    34,  66,  177, 210, 132, 1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   169, 158, 99,  118, 9,   49,  34,  148,
    175, 15,  48,  50,  158, 238, 192, 121, 129, 111, 35,  198, 240, 246, 159,
    196, 103, 149, 207, 12,  223, 180, 245, 157, 130, 0,   103, 220, 171, 217,
    151, 60,  30,  241, 176, 246, 168, 252, 196, 91,  211, 192, 39,  169, 95,
    74,  227, 162, 105, 66,  253, 30,  103, 68,  220, 147, 246, 10,  53,  47,
    157, 64,  10,  241, 80,  81,  142, 177, 3,   100, 79,  137, 34,  210, 16,
    177, 127, 143, 217, 234, 231, 152, 247, 248, 7,   183, 37,  21,  115, 94,
    241, 114, 69,  35,  108, 59,  147, 48,  241, 209, 48,  253, 195, 98,  163,
    86,  2,   242, 247, 40,  255, 139, 223, 248, 71,  107, 143, 125, 199, 43,
    161, 193, 81,  170, 183, 157, 78,  116, 11,  186, 239, 250, 244, 116, 112,
    255, 88,  148, 27,  194, 105, 59,  223, 153, 38,  74,  224, 216, 129, 67,
    35,  8,   189, 85,  55,  24,  84,  79,  78,  95,  78,  100, 246, 214, 36,
    243, 80,  183, 6,   197, 183, 38,  109, 142, 189, 197, 110, 253, 136, 182,
    81,  77,  34,  13,  103, 86,  98,  237, 246, 74,  132, 13,  209, 248, 38,
    178, 227, 16,  167, 146, 70,  201, 194, 251, 202, 156, 255, 230, 58,  100,
    118, 94,  224, 58,  17,  3,   90,  11,  175, 118, 180, 120, 179, 146, 235,
    196, 90,  123, 45,  165, 181, 237, 143, 42,  228, 137, 188, 93,  45,  159,
    100, 240, 189, 115, 162, 81,  171, 15,  115, 94,  241, 114, 69,  35,  108,
    59,  147, 48,  241, 209, 48,  253, 195, 98,  163, 86,  2,   242, 247, 40,
    255, 139, 223, 248, 71,  107, 143, 125, 199, 43,  161, 193, 81,  170, 183,
    157, 78,  116, 11,  186, 239, 250, 244, 116, 112, 255, 88,  148, 27,  194,
    105, 59,  223, 153, 38,  74,  224, 216, 129, 67,  35,  8,   189, 85,  55,
    24,  84,  79,  78,  95,  78,  100, 246, 214, 36,  243, 80,  183, 6,   197,
    183, 38,  109, 142, 189, 197, 110, 253, 136, 182, 81,  77,  34,  13,  103,
    86,  98,  237, 246, 74,  132, 13,  209, 248, 38,  178, 227, 16,  167, 146,
    70,  201, 194, 251, 202, 156, 255, 230, 58,  100, 118, 94,  224, 58,  17,
    3,   90,  11,  175, 118, 180, 120, 179, 146, 235, 196, 90,  123, 45,  165,
    181, 237, 143, 42,  228, 137, 188, 93,  45,  159, 100, 240, 189, 115, 162,
    81,  171, 15,  152, 49,  176, 11,  205, 50,  83,  96,  77,  181, 56,  142,
    42,  36,  138, 246, 55,  247, 141, 105, 239, 78,  64,  63,  205, 200, 253,
    23,  108, 49,  146, 30,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   71,  41,  149, 131, 0,   144, 163, 29,
    1,   20,  175, 36,  56,  179, 196, 28,  229, 5,   87,  110, 13,  35,  17,
    220, 159, 123, 169, 98,  107, 125, 182, 172, 71,  41,  149, 131, 0,   144,
    163, 29,  1,   20,  175, 36,  56,  179, 196, 28,  229, 5,   87,  110, 13,
    35,  17,  220, 159, 123, 169, 98,  107, 125, 182, 172};

constexpr char kPinnedVerifyingKey[] =
    "PinnedVerificationKey { base_modulus: "
    "\"0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47\", "
    "scalar_modulus: "
    "\"0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001\", "
    "domain: PinnedEvaluationDomain { k: 4, extended_k: 6, omega: "
    "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b }, cs: "
    "PinnedConstraintSystem { num_fixed_columns: 1, num_advice_columns: 5, "
    "num_instance_columns: 0, num_selectors: 1, gates: [Product(Product(Fixed "
    "{ query_index: 0, column_index: 0, rotation: Rotation(0) }, Sum(Advice { "
    "query_index: 0, column_index: 0, rotation: Rotation(0) }, Negated(Advice "
    "{ query_index: 1, column_index: 1, rotation: Rotation(0) }))), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))), Product(Fixed { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, "
    "Product(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) }))), Sum(Advice { query_index: 4, column_index: 3, "
    "rotation: Rotation(0) }, Negated(Advice { query_index: 3, column_index: "
    "2, rotation: Rotation(0) })))), Product(Product(Fixed { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))))), Sum(Advice { query_index: 4, column_index: "
    "3, rotation: Rotation(0) }, Negated(Sum(Advice { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, Negated(Advice { query_index: "
    "1, column_index: 1, rotation: Rotation(0) })))))], advice_queries: "
    "[(Column { index: 0, column_type: Advice }, Rotation(0)), (Column { "
    "index: 1, column_type: Advice }, Rotation(0)), (Column { index: 4, "
    "column_type: Advice }, Rotation(0)), (Column { index: 2, column_type: "
    "Advice }, Rotation(0)), (Column { index: 3, column_type: Advice }, "
    "Rotation(0))], instance_queries: [], fixed_queries: [(Column { index: 0, "
    "column_type: Fixed }, Rotation(0))], permutation: Argument { columns: [] "
    "}, lookups: [], constants: [], minimum_degree: None }, fixed_commitments: "
    "[(0x0006d246b1045b5cf7ef706abdca51d7f88992335199a85f594360f81b3435a9, "
    "0x0ce5093fe91a56ef9d54ef73d457aa726be43905202132cd98cb1893cfee96a6)], "
    "permutation: VerifyingKey { commitments: [] } }";

}  // namespace fibonacci3

namespace fibonacci3_v1 {
constexpr char kPinnedConstraintSystem[] =
    "PinnedConstraintSystem { num_fixed_columns: 0, num_advice_columns: 5, "
    "num_instance_columns: 0, num_selectors: 1, gates: "
    "[Product(Product(Selector(Selector(0, true)), Sum(Advice { query_index: "
    "0, column_index: 0, rotation: Rotation(0) }, Negated(Advice { "
    "query_index: 1, column_index: 1, rotation: Rotation(0) }))), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))), Product(Selector(Selector(0, true)), "
    "Product(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) }))), Sum(Advice { query_index: 4, column_index: 3, "
    "rotation: Rotation(0) }, Negated(Advice { query_index: 3, column_index: "
    "2, rotation: Rotation(0) })))), Product(Product(Selector(Selector(0, "
    "true)), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))))), Sum(Advice { query_index: 4, column_index: "
    "3, rotation: Rotation(0) }, Negated(Sum(Advice { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, Negated(Advice { query_index: "
    "1, column_index: 1, rotation: Rotation(0) })))))], advice_queries: "
    "[(Column { index: 0, column_type: Advice }, Rotation(0)), (Column { "
    "index: 1, column_type: Advice }, Rotation(0)), (Column { index: 4, "
    "column_type: Advice }, Rotation(0)), (Column { index: 2, column_type: "
    "Advice }, Rotation(0)), (Column { index: 3, column_type: Advice }, "
    "Rotation(0))], instance_queries: [], fixed_queries: [], permutation: "
    "Argument { columns: [] }, lookups: [], constants: [], minimum_degree: "
    "None }";

constexpr uint8_t kExpectedProof[] = {
    33,  231, 248, 121, 132, 226, 31,  179, 86,  67,  213, 109, 201, 53,  147,
    93,  41,  30,  16,  172, 115, 107, 0,   172, 225, 38,  5,   143, 136, 68,
    3,   47,  222, 110, 120, 246, 167, 202, 30,  64,  192, 113, 187, 20,  209,
    5,   19,  35,  161, 55,  99,  167, 153, 242, 153, 84,  169, 135, 224, 148,
    140, 22,  18,  148, 5,   145, 196, 201, 20,  216, 6,   183, 82,  73,  62,
    87,  36,  44,  76,  29,  120, 103, 124, 253, 87,  100, 218, 122, 245, 84,
    26,  140, 240, 97,  130, 134, 92,  104, 200, 22,  60,  26,  130, 227, 203,
    220, 139, 80,  219, 111, 129, 53,  194, 0,   185, 59,  50,  147, 22,  2,
    25,  58,  255, 70,  194, 103, 39,  5,   143, 199, 14,  63,  41,  37,  111,
    200, 181, 194, 119, 182, 175, 185, 182, 254, 90,  205, 250, 56,  35,  47,
    214, 38,  249, 166, 235, 34,  66,  177, 210, 132, 33,  231, 248, 121, 132,
    226, 31,  179, 86,  67,  213, 109, 201, 53,  147, 93,  41,  30,  16,  172,
    115, 107, 0,   172, 225, 38,  5,   143, 136, 68,  3,   47,  222, 110, 120,
    246, 167, 202, 30,  64,  192, 113, 187, 20,  209, 5,   19,  35,  161, 55,
    99,  167, 153, 242, 153, 84,  169, 135, 224, 148, 140, 22,  18,  148, 5,
    145, 196, 201, 20,  216, 6,   183, 82,  73,  62,  87,  36,  44,  76,  29,
    120, 103, 124, 253, 87,  100, 218, 122, 245, 84,  26,  140, 240, 97,  130,
    134, 92,  104, 200, 22,  60,  26,  130, 227, 203, 220, 139, 80,  219, 111,
    129, 53,  194, 0,   185, 59,  50,  147, 22,  2,   25,  58,  255, 70,  194,
    103, 39,  5,   143, 199, 14,  63,  41,  37,  111, 200, 181, 194, 119, 182,
    175, 185, 182, 254, 90,  205, 250, 56,  35,  47,  214, 38,  249, 166, 235,
    34,  66,  177, 210, 132, 1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   169, 158, 99,  118, 9,   49,  34,  148,
    175, 15,  48,  50,  158, 238, 192, 121, 129, 111, 35,  198, 240, 246, 159,
    196, 103, 149, 207, 12,  223, 180, 245, 157, 130, 0,   103, 220, 171, 217,
    151, 60,  30,  241, 176, 246, 168, 252, 196, 91,  211, 192, 39,  169, 95,
    74,  227, 162, 105, 66,  253, 30,  103, 68,  220, 147, 246, 10,  53,  47,
    157, 64,  10,  241, 80,  81,  142, 177, 3,   100, 79,  137, 34,  210, 16,
    177, 127, 143, 217, 234, 231, 152, 247, 248, 7,   183, 37,  21,  115, 94,
    241, 114, 69,  35,  108, 59,  147, 48,  241, 209, 48,  253, 195, 98,  163,
    86,  2,   242, 247, 40,  255, 139, 223, 248, 71,  107, 143, 125, 199, 43,
    161, 193, 81,  170, 183, 157, 78,  116, 11,  186, 239, 250, 244, 116, 112,
    255, 88,  148, 27,  194, 105, 59,  223, 153, 38,  74,  224, 216, 129, 67,
    35,  8,   189, 85,  55,  24,  84,  79,  78,  95,  78,  100, 246, 214, 36,
    243, 80,  183, 6,   197, 183, 38,  109, 142, 189, 197, 110, 253, 136, 182,
    81,  77,  34,  13,  103, 86,  98,  237, 246, 74,  132, 13,  209, 248, 38,
    178, 227, 16,  167, 146, 70,  201, 194, 251, 202, 156, 255, 230, 58,  100,
    118, 94,  224, 58,  17,  3,   90,  11,  175, 118, 180, 120, 179, 146, 235,
    196, 90,  123, 45,  165, 181, 237, 143, 42,  228, 137, 188, 93,  45,  159,
    100, 240, 189, 115, 162, 81,  171, 15,  115, 94,  241, 114, 69,  35,  108,
    59,  147, 48,  241, 209, 48,  253, 195, 98,  163, 86,  2,   242, 247, 40,
    255, 139, 223, 248, 71,  107, 143, 125, 199, 43,  161, 193, 81,  170, 183,
    157, 78,  116, 11,  186, 239, 250, 244, 116, 112, 255, 88,  148, 27,  194,
    105, 59,  223, 153, 38,  74,  224, 216, 129, 67,  35,  8,   189, 85,  55,
    24,  84,  79,  78,  95,  78,  100, 246, 214, 36,  243, 80,  183, 6,   197,
    183, 38,  109, 142, 189, 197, 110, 253, 136, 182, 81,  77,  34,  13,  103,
    86,  98,  237, 246, 74,  132, 13,  209, 248, 38,  178, 227, 16,  167, 146,
    70,  201, 194, 251, 202, 156, 255, 230, 58,  100, 118, 94,  224, 58,  17,
    3,   90,  11,  175, 118, 180, 120, 179, 146, 235, 196, 90,  123, 45,  165,
    181, 237, 143, 42,  228, 137, 188, 93,  45,  159, 100, 240, 189, 115, 162,
    81,  171, 15,  152, 49,  176, 11,  205, 50,  83,  96,  77,  181, 56,  142,
    42,  36,  138, 246, 55,  247, 141, 105, 239, 78,  64,  63,  205, 200, 253,
    23,  108, 49,  146, 30,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   71,  41,  149, 131, 0,   144, 163, 29,
    1,   20,  175, 36,  56,  179, 196, 28,  229, 5,   87,  110, 13,  35,  17,
    220, 159, 123, 169, 98,  107, 125, 182, 172, 71,  41,  149, 131, 0,   144,
    163, 29,  1,   20,  175, 36,  56,  179, 196, 28,  229, 5,   87,  110, 13,
    35,  17,  220, 159, 123, 169, 98,  107, 125, 182, 172};

constexpr char kPinnedVerifyingKey[] =
    "PinnedVerificationKey { base_modulus: "
    "\"0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47\", "
    "scalar_modulus: "
    "\"0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001\", "
    "domain: PinnedEvaluationDomain { k: 4, extended_k: 6, omega: "
    "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b }, cs: "
    "PinnedConstraintSystem { num_fixed_columns: 1, num_advice_columns: 5, "
    "num_instance_columns: 0, num_selectors: 1, gates: [Product(Product(Fixed "
    "{ query_index: 0, column_index: 0, rotation: Rotation(0) }, Sum(Advice { "
    "query_index: 0, column_index: 0, rotation: Rotation(0) }, Negated(Advice "
    "{ query_index: 1, column_index: 1, rotation: Rotation(0) }))), "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))), Product(Fixed { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, "
    "Product(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) }))), Sum(Advice { query_index: 4, column_index: 3, "
    "rotation: Rotation(0) }, Negated(Advice { query_index: 3, column_index: "
    "2, rotation: Rotation(0) })))), Product(Product(Fixed { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, "
    "Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Sum(Constant("
    "0x0000000000000000000000000000000000000000000000000000000000000001), "
    "Negated(Product(Sum(Advice { query_index: 0, column_index: 0, rotation: "
    "Rotation(0) }, Negated(Advice { query_index: 1, column_index: 1, "
    "rotation: Rotation(0) })), Advice { query_index: 2, column_index: 4, "
    "rotation: Rotation(0) })))))), Sum(Advice { query_index: 4, column_index: "
    "3, rotation: Rotation(0) }, Negated(Sum(Advice { query_index: 0, "
    "column_index: 0, rotation: Rotation(0) }, Negated(Advice { query_index: "
    "1, column_index: 1, rotation: Rotation(0) })))))], advice_queries: "
    "[(Column { index: 0, column_type: Advice }, Rotation(0)), (Column { "
    "index: 1, column_type: Advice }, Rotation(0)), (Column { index: 4, "
    "column_type: Advice }, Rotation(0)), (Column { index: 2, column_type: "
    "Advice }, Rotation(0)), (Column { index: 3, column_type: Advice }, "
    "Rotation(0))], instance_queries: [], fixed_queries: [(Column { index: 0, "
    "column_type: Fixed }, Rotation(0))], permutation: Argument { columns: [] "
    "}, lookups: [], constants: [], minimum_degree: None }, fixed_commitments: "
    "[(0x0006d246b1045b5cf7ef706abdca51d7f88992335199a85f594360f81b3435a9, "
    "0x0ce5093fe91a56ef9d54ef73d457aa726be43905202132cd98cb1893cfee96a6)], "
    "permutation: VerifyingKey { commitments: [] } }";

}  // namespace fibonacci3_v1
}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI3_CIRCUIT_TEST_DATA_H_
