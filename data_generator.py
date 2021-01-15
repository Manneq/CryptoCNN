"""
    File 'data_generator.py' consists of methods for model training.
"""
import numpy as np
import onehot


def image_generator(image_size):
    """
        Method to generate random image for model training.
        param:
            image_size - size of image
        return:
            Generated image.
    """
    return np.random.randint(0, 256, image_size) / 255.


def message_generator(message_length, dictionary_length):
    """
        Method to generate random message for model training.
        param:
            1. message_length - maximum length of sentence
            2. dictionary_length - size of the dictionary
        return:
            Generated sentence.
    """
    return np.array([np.random.randint(0, dictionary_length, message_length)])


def training_data_generator(image_size, message_length, dictionary_length,
                            batch_size=32):
    """
        Method to generate batches for model training.
        param:
            1. image_size - size of the image
            2. message_length - maximum length of image
            3. dictionary_length - size of the dictionary
            4. batch_size - size of the training batch
        return:
            Batch of the random data for training
    """
    while True:
        # Input data initialization
        x_image = np.zeros((batch_size, image_size[0], image_size[1],
                            image_size[2]))
        x_sentence = np.zeros((batch_size, message_length))
        # Output data initialization
        y_image = np.zeros((batch_size, image_size[0], image_size[1],
                            image_size[2]))
        y_sentence = np.zeros((batch_size, message_length,
                               dictionary_length))

        for i in range(batch_size):
            # Input and output data generation
            image = image_generator(image_size)
            sentence = message_generator(message_length, dictionary_length)
            sentence_onehot = onehot.onehot_encoder(sentence,
                                                    dictionary_length)

            x_image[i] = image
            x_sentence[i] = sentence
            y_image[i] = image
            y_sentence[i] = sentence_onehot

        yield [[x_image, x_sentence], [y_image, y_sentence]]
