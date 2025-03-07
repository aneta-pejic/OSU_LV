
words_num = {}

fhand = open('song.txt')

for line in fhand:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word in words:
            words_num[word.lower()] += 1

print(words_num)

fhand.close () 