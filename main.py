from utils.file_io import load_EMG_data, load_optdigits_data
from utils.shuffle import shuffle

def main():
    (X, Y) = load_optdigits_data(5000)
    (X, Y) = shuffle(X, Y)
    print('Hello Project4')

if __name__ == "__main__":
    main()