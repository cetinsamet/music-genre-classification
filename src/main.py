import matplotlib.pyplot as plt

from config import GENRES
from config import DATAPATH
from data import Data
from set import Set


def main():
    # ------------------------------------------------------------------------------------------- #
    ## DATA
    #data    = Data(GENRES, DATAPATH)
    #data.make_raw_data()
    #data.save()
    data    = Data(GENRES, DATAPATH)
    data.load()
    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    ## SET
    set_    = Set(data)
    set_.make_dataset()
    x_train, y_train    = set_.get_train_set()
    x_valid, y_valid    = set_.get_valid_set()
    x_test,  y_test     = set_.get_test_set()
    # ------------------------------------------------------------------------------------------- #


    '''
    plt.figure(figsize=(10, 4))
    specshow(power_to_db(data_chunks[0]))
    plt.colorbar(format='%+2.0f dB')
    plt.title(TRACKPATH + ' spectrogram')
    plt.tight_layout()
    plt.show()
    '''


if __name__ == '__main__':
    main()





