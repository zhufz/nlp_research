import sys
if __name__ == '__main__':
    train_file_path = sys.argv[1]
    vocab = set()
    n_train_tokens = 0
    with open(train_file_path) as f_in:
        for line in f_in:
            line = line.strip().split()
            n_train_tokens += len(line)
            for word in line:
                vocab.add(word)

    with open("data/vocab.txt",'w') as f_out:
        for word in list(vocab):
            f_out.write(word+"\n")
        f_out.write("<S>\n")
        f_out.write("</S>\n")
        f_out.write("<UNK>\n")
    print("n_train_tokens:%s"%n_train_tokens)


