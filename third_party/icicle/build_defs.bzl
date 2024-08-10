BN254 = "bn254"
BLS12_381 = "bls12_381"
BLS12_377 = "bls12_377"
BW6_761 = "bw6_761"
GRUMPKIN = "grumpkin"
BABY_BEAR = "baby_bear"
STARK_252 = "stark_252"
M31 = "m31"

CURVES = [
    BN254,
    BLS12_381,
    BLS12_377,
    BW6_761,
    GRUMPKIN,
]

FIELDS = [
    BN254,
    BLS12_381,
    BLS12_377,
    BW6_761,
    GRUMPKIN,
    BABY_BEAR,
    STARK_252,
    M31,
]

FIELDS_WITH_MMCS = FIELDS

FIELDS_WITH_NTT = [
    BN254,
    BLS12_381,
    BLS12_377,
    BW6_761,
    BABY_BEAR,
    STARK_252,
]

FIELDS_WITH_POSEIDON = CURVES

FIELDS_WITH_POSEIDON2 = [
    BN254,
    BABY_BEAR,
]

def icicle_defines(field):
    if field == BN254:
        return [
            "FIELD_ID=BN254",
            "CURVE_ID=BN254",
            "CURVE=bn254",
        ]
    elif field == BLS12_381:
        return [
            "FIELD_ID=BLS12_381",
            "CURVE_ID=BLS12_381",
            "CURVE=bls12_381",
        ]
    elif field == BLS12_377:
        return [
            "FIELD_ID=BLS12_377",
            "CURVE_ID=BLS12_377",
            "CURVE=bls12_377",
        ]
    elif field == BW6_761:
        return [
            "FIELD_ID=BW6_761",
            "CURVE_ID=BW6_761",
            "CURVE=bw6_761",
        ]
    elif field == GRUMPKIN:
        return [
            "FIELD_ID=GRUMPKIN",
            "CURVE_ID=GRUMPKIN",
            "CURVE=grumpkin",
        ]
    elif field == BABY_BEAR:
        return ["FIELD_ID=BABY_BEAR"]
    elif field == STARK_252:
        return ["FIELD_ID=STARK_252"]
    return []
