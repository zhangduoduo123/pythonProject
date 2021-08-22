import collections


def firstUniqChar( s: str) :
    frequency = collections.Counter(s)
    for i, ch in enumerate(s):
        if frequency[ch] == 1:
            return i
    return -1


print(firstUniqChar('loveleetcode'))