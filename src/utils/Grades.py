GRADES = {
    "4a" : 10,
    "4b" : 11,
    "4c" : 12,
    "5a" : 13,
    "5b" : 14,
    "5c" : 15,
    "6a" : 16,
    "6a+" : 17,
    "6b" : 18,
    "6b+" : 19,
    "6c" : 20,
    "6c+" : 21,
    "7a" : 22,
    "7a+" : 23,
    "7b" : 24,
    "7b+" : 25,
    "7c" : 26,
    "7c+" : 27,
    "8a" : 28,
    "8a+" : 29,
    "8b" : 30,
    "8b+" : 31,
    "8c" : 32,
    "8c+" : 33
}

def grade_label(value):
    """The GRADES label (e.g. "7a") whose numeric value is closest to `value`
    - for turning a predicted/generated numeric grade back into something readable."""
    return min(GRADES.items(), key=lambda kv: abs(kv[1] - value))[0]