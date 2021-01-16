"""
    File 'data_management.py' consists of methods to import and export data.
"""
import numpy as np
import cv2
import os
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


def messages_from_text_loading(path_to_file, message_length, messages_number):
    """
        Method to load messages from text file (usable only for evaluation).
        param:
            1. path_to_file - string path to text file
            2. message_length - maximum length of message
            3. messages_number - number of messages to import
        return:
            messages - list of ascii encoded messages
    """
    file = open(path_to_file, 'r')

    messages = [ascii.ascii_encoder(message.split("\n")[0], message_length)[0]
                for message in file.readlines()[:messages_number]]

    file.close()

    return messages


def images_names_loading(path_to_folder,
                         message_length):
    """
        Method to load list of images (usable only for evaluation).
        param:
            path_to_folder - string path to folder with images
        return:
            List of images
    """
    return [image_loading(path_to_folder + "/" + file, message_length)[0]
            for file in os.listdir(path_to_folder)]
