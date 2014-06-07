import sys, os, csv

# Extract quantifier, noun, verb from string with format "( q n ) v"
def extract_words(phrase):
    tokens = phrase.split()
    return (tokens[1], tokens[2], tokens[4])
    
# Generates entailment data in which every unique entailment relationship
# in the dataset where the noun and verb on either side are the same is
# represented. Eg. | (two hungry) mobile (no hungry) mobile -> | two no
def generate_data(in_folder, out_file):
    relns = set()
    print "Scanning data..."
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            if not name.startswith("NEG") and name.endswith("tsv"): # ignore NEG files
                with open(os.path.join(root, name)) as f:
                    for line in csv.reader(f, delimiter="\t"):
                        reln = line[0]
                        left = extract_words(line[1])
                        right = extract_words(line[2])
                        if left[1] == right[1] and left[2] == right[2]:
                            # only add if nouns and verbs identical
                            relns.add((reln, left[0], right[0]))

    #write relns to .tsv file
    with open(out_file, 'w') as out:
        for reln in relns:
            out.write('{0}\t{1}\t{2}\n'.format(reln[0], reln[1], reln[2]))

if __name__ == "__main__":

    in_folder = sys.argv[1]
    out_file = sys.argv[2]
    
    generate_data(in_folder, out_file)
    
    

    
