import sys

path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

with open(path, "r", encoding="utf-8") as handle:
    for index, line in enumerate(handle, start=1):
        if start <= index <= end:
            print(f"{index}: {line.rstrip()}")
