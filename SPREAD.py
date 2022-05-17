import re
import argparse
import numpy as np
from Bio import SeqIO
from sklearn.externals import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model




def extract_seq(fasta_fname):
    """
    param fasta_fname: Fasta file address.

    """
    seq_list = []
    seq_id = []

    for seq_record in SeqIO.parse(fasta_fname, 'fasta'):

        bool = re.search(r'[^ATCGN]',(str(seq_record.seq)).upper())
        if bool:continue

        seq = (str(seq_record.seq)).upper()
        id = str(seq_record.description)

        seq_list.append(seq)
        seq_id.append(id)
    return seq_list, seq_id


protein_encoding = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4
}


def encode(line):
    length = len(line)
    encoded = []

    for i in range(0, length):
        encoded.append(protein_encoding[line[i]])
    return encoded


def data(seq_list):
    Data=[]

    for line in seq_list:
        line_new = line.strip()
        line_encode = encode(line_new)
        Data.append(line_encode)
    return Data


def calculate_performace(y_pred):
    labels = []

    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            labels.append(1)
        if y_pred[i] < 0.5:
            labels.append(0)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='SPREAD:recognizing promoters in Pseudomonas aeruginosa.')
    parser.add_argument('--input', dest='inputfile', type=str, required=True,
                        help='query sequences to be predicted in fasta format.')
    parser.add_argument('--output', dest='outputfile', type=str, required=False,
                        help='save the prediction results.')
    args = parser.parse_args()

    inputfile = args.inputfile
    outputfile = args.outputfile

    outputfile_original = outputfile
    if outputfile_original == None:
        outputfile = 'output.txt'

    fasta_fname = inputfile
    seq_list, seq_id = extract_seq(fasta_fname)
    Data = np.array(data(seq_list))
    Data = to_categorical(Data)

    Data_shape = Data.shape
    if Data_shape[2] == 4:
        zeros = np.zeros((Data_shape[0], Data_shape[1], 1))
        Data = np.concatenate((Data, zeros), axis=2)

    model_1 = load_model('model/lstm_autoencoder_16.h5')
    model_2 = load_model('model/lstm_autoencoder_512.h5')

    latent_vector_model_1 = Model(inputs=model_1.input, outputs=model_1.get_layer('lstm_1').output)
    latent_vector_model_2 = Model(inputs=model_2.input, outputs=model_2.get_layer('lstm_1').output)

    x_1 = latent_vector_model_1.predict(Data)
    x_2 = latent_vector_model_2.predict(Data)
    x_2 = x_2[:,80,:]

    cnn = load_model('model/cnn_fold_10-16.h5')

    y_predict = cnn.predict(x_1)
    sdata = y_predict.shape
    y_predict1 = np.reshape(y_predict, [sdata[0]])

    RF = joblib.load('model/rf_fold_10-512.m')

    y_predict2 = RF.predict_proba(x_2)
    y_predict2 = y_predict2[:, 1]

    y_pred = (np.array(y_predict1) + np.array(y_predict2)) / 2

    labels = calculate_performace(y_pred)

    with open(outputfile, 'w') as f:
        for i in range(len(seq_id)):
            f.write(seq_id[i] + '\n')
            f.write(str(labels[i]) + '\n')
    print('output are saved in ' + outputfile + ', and those identified as promoters are marked with *')


if __name__ == "__main__":
    main()