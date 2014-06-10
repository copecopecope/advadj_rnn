import sys, os

if __name__=="__main__":
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    minPairs = int(sys.argv[3])

    print 'Scanning file...'
    with open(inFile, 'r') as inF:
        lines = inF.readlines()

    print 'Culling to adv-adj pairs in {0}...'.format(outFile)
    with open(outFile, 'w') as outF:
        i = 1
        while i < len(lines):
            if i % 1000 == 1:
                sys.stdout.write('.')
            line = lines[i].split(',')
            if line[1] == 'r' and line[3] == 'a':
                # write to file if total dist sum > minPairs
                totalCount = 0
                counts = [0]*10
                adv = line[0]
                adj = line[2]
                for j in range(i,i+10):
                    l = lines[j].split(',')
                    count = int(l[5])
                    counts[j-i] = count
                    totalCount += count
                if totalCount >= minPairs:
                    outF.write("{0},{1}".format(adv,adj))
                    for ct in counts:
                        outF.write(",{0}".format(ct))
                    outF.write("\n")
            i += 10
    sys.stdout.write('\n')
            
            
