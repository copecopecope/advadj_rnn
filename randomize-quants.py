import random

if __name__=="__main__":
    quants = ['=', '>', '<', '#', '|', '^']
    with open("quant-data.tsv", "r") as f:
        with open("randomized-quant-data.tsv", "w") as out:
            for line in f.readlines():
                rand_quant = random.sample(quants, 1)[0]
                line = rand_quant + line[1:]
                out.write(line)
    
