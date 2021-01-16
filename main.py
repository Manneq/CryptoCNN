"""
    File 'main.py' is main file that controls the sequence of functions calls.
"""
import data_management
import neural_network
import plotting


def encryption(encoder, message, path_to_image, key_1, key_2, message_length,
               dictionary_length):
    """
        Method to encrypt message into image.
        param:
            1. encoder - encoder part of the CNN
            2. message - message that need to be encoded
            3. path_to_image - string path to image
            4. key_1 - first key for Affine cipher
            5. key_2 - second ket for Affine cipher
            6. message_length - maximum length of the message
            7. dictionary_length - length of the dictionary
    """
    # Image loading
    image = data_management.image_loading(path_to_image, message_length)
    # Message loading and encryption using Affine cipher
    message = \
        data_management.message_loading_and_encryption(message, key_1, key_2,
                                                       message_length,
                                                       dictionary_length)

    # Message encryption into image
    image = neural_network.model_encoding(encoder, image, message)

    # Image saving
    data_management.image_saving(path_to_image, image)

    return


def decryption(decoder, path_to_image, key_1, key_2, message_length,
               dictionary_length):
    """
        Method to encrypt message into image.
        param:
            1. decoder - decoder part of the CNN
            2. message - message that need to be encoded
            3. path_to_image - string path to image
            4. key_1 - first key for Affine cipher
            5. key_2 - second ket for Affine cipher
            6. message_length - maximum length of the message
            7. dictionary_length - length of the dictionary
    """
    # Image loading
    image = data_management.image_loading(path_to_image, message_length)

    # Message extraction
    message = neural_network.model_decoding(decoder, image)

    # Message decryption using Affine cipher and showing
    data_management.message_decryption_and_showing(message, key_1, key_2,
                                                   dictionary_length)

    return


def test(encoder, decoder, message, path_to_image, key_1, key_2,
         message_length, dictionary_length):
    """
        Method to test model in one function.
        param:
            1. encoder - encoder part of the CNN
            2. decoder - decoder part of the CNN
            3. message - message that need to be encoded
            4. path_to_image - string path to image
            5. key_1 - first key for Affine cipher
            6. key_2 - second ket for Affine cipher
            7. message_length - maximum length of the message
            8. dictionary_length - length of the dictionary
    """
    # Image loading
    image = data_management.image_loading(path_to_image, message_length)
    # Message loading and encryption using Affine cipher
    message = \
        data_management.message_loading_and_encryption(message, key_1, key_2,
                                                       message_length,
                                                       dictionary_length)

    # Encoding and decoding message
    image, message = neural_network.model_test(encoder, decoder, image,
                                               message)

    # Image saving
    data_management.image_saving(path_to_image, image)
    # Message decryption using Affine cipher and showing
    data_management.message_decryption_and_showing(message, key_1, key_2,
                                                   dictionary_length)

    return


def main():
    """
        Main function.
    """
    # Message length
    message_length = 100
    # Image size
    image_size = (message_length, message_length, 3)
    # Dictionary length (Ascii table)
    dictionary_length = 128
    # Keys for Affine cypher
    key_1, key_2 = 7, 21

    """
    # Model training
    neural_network.model_training(image_size, sentence_length,
                                  dictionary_length)"""
    """
    # Model evaluation
    neural_network.model_evaluation("data/test_set",
                                    "data/words.txt",
                                    image_size,
                                    message_length,
                                    dictionary_length)"""

    # Model loading
    model, encoder, decoder = neural_network.model_loading(image_size,
                                                           message_length,
                                                           dictionary_length)

    # Models plotting
    plotting.model_plotting(model, "plots/full_model.png")
    plotting.model_plotting(encoder, "plots/encoder_model.png")
    plotting.model_plotting(decoder, "plots/decoder_model.png")

    # Model testing
    test(encoder, decoder, "Neo-Doomer", "data/doomer.jpg", key_1,
         key_2, message_length, dictionary_length)


    return


if __name__ == "__main__":
    main()
