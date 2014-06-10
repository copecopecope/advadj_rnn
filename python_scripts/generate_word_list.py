import sys, os

if __name__=="__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]

    print 'Scanning file...'
    with open(inFile, 'r') as inF:
        lines = inF.readlines()

    print 'Creating word list in {0}...'.format(outFile)
    wordList = set()
    for line in lines:
        line = line.split(',')
        wordList.add(line[0])
        wordList.add(line[1])
    with open(outFile, 'w') as outF:
        for word in wordList:
            outF.write(word)
            outF.write('\n')
                
            
