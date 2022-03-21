def binaryStrToInt(b):
    result = 0
    n = len(b)
    for i in range(len(b) - 1, -1, -1):
        if b[i] == '1':
            result += 2**(n - i - 1)
    return result

print(binaryStrToInt('101'))
print(binaryStrToInt('111'))
print(binaryStrToInt('011100'))