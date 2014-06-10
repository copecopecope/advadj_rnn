import sys

if __name__=="__main__":

    testFile = "data/advadj_test.csv"
    testResultsFile = sys.argv[1]

    with open(testFile, 'r') as f:
        testLines = f.readlines()

    with open(testResultsFile, 'r') as f:
        trLines = f.readlines()

    assert(len(testLines) == len(trLines))

    
