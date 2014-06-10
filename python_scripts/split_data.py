import sys, random, math

if __name__=="__main__":

    inFile = sys.argv[1]
    outTrain = inFile.replace(".csv", "_train.csv")
    outTest = inFile.replace(".csv", "_test.csv")

    SPLIT_PERCENT = 0.70
    
    with open(inFile) as f:
        lines = f.readlines()

    random.shuffle(lines)

    endOfTrain = int(math.ceil(len(lines)*SPLIT_PERCENT))

    with open(outTrain, 'w') as f:
        for i in range(endOfTrain):
            f.write(lines[i])

    with open(outTest, 'w') as f:
        for i in range(endOfTrain,len(lines)):
            f.write(lines[i])
