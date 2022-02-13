verboseLevel = 0


def setVerboseLevel(level):
    global verboseLevel
    verboseLevel = level


def verbosePrint(message: str, minVerbosity: int):
    if verboseLevel >= minVerbosity:
        print("    " * (minVerbosity - 1), end="")
        print(message)
