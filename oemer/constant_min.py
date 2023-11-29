# Legend:
#   0: background
#   2: ledgerline
#   3: barline between
#   4: barline end
#  10: g-glef
#  11: c-clef
#  12: c-clef again
#  13: f-clef
#  19: numbers - ignore
#  24: time-signature 3
#  28: time-signature 7
#  34: time-signature cut time
#  35: notehead full on line
#  36: Unknown
#  37: notehead full between lines
#  38: Unknown
#  39: notehead hollow on line
#  40: Unknown
#  41: notehead hollow between line
#  43: whole-note on line
#  45: hole-note between line
#  47: double-whole-note on line
#  49: double-whole-note between line
#  51: dot between
#  52: stem
#  53: tremolo 1
#  54: tremolo 2 
#  55: tremolo 3
#  56: tremolo 4
#  58: flag 1 down
#  60: flag 2 down
#  61: flag 3 down
#  62: flag 4 down
#  63: flag 5 down
#  64: flag 1 up
#  66: flag 2 up
#  67: flag 3 up
#  68: flag 4 up
#  69: flag 5 up
#  70: flat
#  72: natural
#  74: sharp
#  76: double sharp
#  78: key flat
#  79: key natural
#  80: key sharp
#  81: accent above
#  82: accent below
#  83: staccato above
#  84: staccato below
#  85: tenuto above
#  86: tenuto below
#  87: Staccatissimo above
#  88: Staccatissimo below
#  89: marcato above
#  90: marcato below
#  91: fermata above
#  92: fermata below
#  93: breath mark
#  95: rest large
#  96: rest long
#  97: rest breve
#  98: rest full
#  99: rest quarter
# 100: rest eigth
# 101: rest 16th
# 102: rest 32th
# 103: rest 64th
# 104: rest 128th
# 127: trill
# 128: unknown
# 129: gruppeto
# 130: mordent
# 131: down-bow
# 132: up-bow
# 133: symbol
# 134: symbol
# 135: symbol
# 136: symbol
# 137: symbol
# 138: symbol
# 139: symbol
# 141: symbol
# 142: symbol
# 143: unknown
# 144: unknown
# 145: slur
# 146: beam
# 147: slur
# 148: unknown
# 149: unknown
# 150: unknown
# 151: unknown
# 152: unknown
# 153: unknown
# 154: unknown
# 155: unknown
# 156: unknown
# 157: unknown
# 159: unknown
# 160: unknown
# 161: unknown
# 162: unknown
# 163: unknown
# 164: unknown
# 165: staff
# 167: unknown
# 170: unknown
# 171: unknown

CLASS_CHANNEL_LIST = [
    [165, 2],  # staff, ledgerLine
    [35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 52],  # notehead, stem
    [
        64, 58, 60, 66, 63, 69, 68, 61, 62, 67, 65, 59, 146,  # flags, beam
        97, 100, 99, 98, 101, 102, 103, 104, 96, 163,  # rests
        80, 78, 79, 74, 70, 72, 76, 3,  # sharp, flat, natural, barline
        10, 13, 12, 19, 11, 20, 51, # clefs, augmentationDot, 
        25, 24, 29, 22, 23, 28, 27, 34, 30, 21, 33, 26,  # timeSigs
    ]
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2
