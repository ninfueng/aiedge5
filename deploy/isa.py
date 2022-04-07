import os

from typing import List


REGMAP = {
    "zero": 0,
    "ra": 1,
    "sp": 2,
    "gp": 3,
    "tp": 4,
    "t0": 5,
    "t1": 6,
    "t2": 7,
    "s0": 8,
    "fp": 8,
    "s1": 9,
    "a0": 10,
    "a1": 11,
    "a2": 12,
    "a3": 13,
    "a4": 14,
    "a5": 15,
    "a6": 16,
    "a7": 17,
    "s2": 18,
    "s3": 19,
    "s4": 20,
    "s5": 21,
    "s6": 22,
    "s7": 23,
    "s8": 24,
    "s9": 25,
    "s10": 26,
    "s11": 27,
    "t3": 28,
    "t4": 29,
    "t5": 30,
    "t6": 31,
}


def addi(rd, rs1, imm):
    rd = bin(rd)[2:].zfill(5)
    rs1 = bin(rs1)[2:].zfill(5)
    imm = bin(imm)[2:].zfill(12)
    opcode = "0010011"
    funct3 = "000"
    return hex(int(imm + rs1 + funct3 + rd + opcode, 2))


def lui(rd: int, imm: int) -> int:
    """U-type instruction"""
    assert rd in range(0, 31)

    opcode = "0110111"
    rd = bin(rd)[2:].zfill(5)
    imm = bin(imm)[2:].zfill(32)[:20]
    instruct = imm + rd + opcode
    return int(instruct, 2)


def lw(rd: int, rs1: int, imm: int) -> int:
    """I-type instruction"""
    assert rd in range(0, 31)
    assert rs1 in range(0, 31)

    opcode = "0000011"
    funct3 = "010"
    rs1 = bin(rs1)[2:].zfill(5)
    rd = bin(rd)[2:].zfill(5)
    imm = bin(imm)[2:].zfill(12)
    instruct = imm + rs1 + funct3 + rd + opcode
    return int(instruct, 2)


def sw(rs1: int, rs2: int, imm: int) -> int:
    """S-type instruction"""
    assert rs1 in range(0, 31)
    assert rs2 in range(0, 31)

    opcode = "0100011"
    funct3 = "010"
    rs1 = bin(rs1)[2:].zfill(5)
    rs2 = bin(rs2)[2:].zfill(5)
    imm = bin(imm)[2:].zfill(12)
    instruct = imm[:7] + rs2 + rs1 + funct3 + imm[len(imm) - 5 : len(imm)] + opcode
    return int(instruct, 2)


def load_instructs(dump_dir: str) -> List[int]:
    dump_dir = os.path.expanduser(dump_dir)
    instructs = []
    with open(dump_dir, "r") as f:
        for line in f:
            line = line.strip()
            instructs.append(int(line, 16))
    return instructs


if __name__ == "__main__":
    test = lui(8, 0xA0002000)
    print(hex(test))

    test = lw(12, 8, 0x0)
    print(hex(test))

    test = lw(13, 8, 0x4)
    print(hex(test))

    test = sw(8, 14, 0x8)
    print(hex(test))

    test = lui(8, 0xA0020000)
    print(hex(test))
