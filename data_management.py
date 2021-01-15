"""
    File 'data_management.py' consists of methods to import and export data.
"""
import numpy as np
import cv2
import affine
import ascii
import onehot


def image_loading(path_to_image, message_length):
    """
        Method to load image.
        param:
            1. path_to_image - string path to image
            2. message_length -  maximum length of message
        return:
            Image that can be passed to the model
    """
    return np.expand_dims(
        cv2.resize(cv2.imread(path_to_image, 1),
                   (message_length, message_length)) / 255., axis=0)


def image_saving(path_to_image, image):
    """
        Method to save image.
        param:
            1. path_to_image - string path to image
            2. image -
    """
    cv2.imwrite("/".join(path_to_image.split("/")[:-1]) + "/image_encoded.png",
                image * 255.)

    return


def message_loading_and_encryption(message, key_1, key_2, message_length,
                                   dictionary_length):
    """
        Method to preprocess message.
        param:
            1. message - message that need to be encrypted
            2. key_1 - first key for Affine cipher
            3. key_2 - second key for Affine cipher
            4. message_length - maximum length of message
            5. dictionary_length - size of the dictionary
        return:
            message - message converted into ascii codes
    """
    # Affine encryption
    message = affine.affine_encryption(message, key_1, key_2,
                                       dictionary_length)

    # Conversion into ascii codes
    message = ascii.ascii_encoder(message, message_length)

    return message


def message_decryption_and_showing(message, key_1, key_2,
                                   dictionary_length):
    """
        Method to postprocess message.
        param:
            1. message - decrypted message from model
            2. key_1 - first key for Affine cipher
            3. key_2 - second key for Affine cipher
            4. dictionary_length - size of the dictionary
        return:
            message - message converted into ascii codes
    """
    # Decoding message using onehot into ascii codes
    message = onehot.onehot_decoder(message)

    # Ascii codes into string message conversion
    message = ascii.ascii_decoder(message)

    # Message decryption using Affine cipher and showing
    print(affine.affine_decryption(message, key_1, key_2,
                                   dictionary_length))

    return
