"""
    File 'onehot.py' consists of methods to encode and decode ascii message
        using Onehot method.
"""
import numpy as np


def onehot_encoder(ascii_message, dictionary_length):
    """
        Method to encode ascii codes using onehot method.
        param:
            1. ascii_message - ascii encoded message
            2. dictionary_length - length of the dictionary
        return:
             Onehot encoded message
    """
    return np.eye(dictionary_length)[ascii_message.reshape(-1)]


def onehot_decoder(onehot_message):
    """
        Method to decode message into ascii codes from onehot.
        param:
            onehot_message - onehot encoded message
        return:
            Ascii encoded message
    """
    return np.argmax(onehot_message, axis=1)
