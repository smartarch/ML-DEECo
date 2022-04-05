verboseLevel = 0
file = None


def setVerboseLevel(level):
    global verboseLevel
    verboseLevel = level


def setVerbosePrintFile(outputFile):
    global file
    file = outputFile


def verbosePrint(message: str, minVerbosity: int):
    if verboseLevel >= minVerbosity:
        print("    " * (minVerbosity - 1), end="")
        print(message)
        if file:
            print("    " * (minVerbosity - 1), end="", file=file)
            print(message, file=file)
