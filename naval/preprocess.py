import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    for line in f:
        if not line.startswith('@'):
            print(line)
