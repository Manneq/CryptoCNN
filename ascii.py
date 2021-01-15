"""
    File 'ascii.py' consists of functions to encode message into ascii codes
        or decode from it.
"""
import numpy as np


def ascii_encoder(message, message_length):
    """
        Method to encode message into ascii codes.
        param:
            1. message - string message that need to be encoded
            2. message_length - maximum length of message
        return:
            ascii_message - message encoded into ascii codes
    """
    ascii_message = np.zeros((1, message_length))

    for i, symbol in enumerate(message.encode("ascii")):
        ascii_message[0, i] = symbol

    return ascii_message


def ascii_decoder(ascii_message):
    """
        Method to convert ascii codes into string message.
        param:
            ascii_message - message incoded into ascii codes
        return:
            String message
    """
    return ''.join(chr(int(symbol_code)) for symbol_code in ascii_message
                   if symbol_code != 0)
