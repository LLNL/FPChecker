#!/usr/bin/env python3

# Generator
# Yield successive n-sized chunks from lst.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    x = [1,4,6,7,8,9,23,25,56,55,57,68,69,70]
    print(x, len(x))
    for i in chunks(x, 2):
        print(str(i))

if __name__ == '__main__':
    main()
