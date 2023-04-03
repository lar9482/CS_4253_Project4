from utils.file_io import load_EMG_data, load_optdigits_data, load_spambase_data
from utils.shuffle import shuffle

def main():
    (X, Y) = load_spambase_data(4500)
    (X, Y) = shuffle(X, Y)
    print(Y)
    print('Hello Project4')

if __name__ == "__main__":
    main()