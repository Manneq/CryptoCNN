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
        input_images = np.zeros((batch_size, image_size[0], image_size[1],
                                 image_size[2]))
        input_messages = np.zeros((batch_size, message_length))
        # Output data initialization
        output_images = np.zeros((batch_size, image_size[0], image_size[1],
                                 image_size[2]))
        output_messages = np.zeros((batch_size, message_length,
                                   dictionary_length))

        for i in range(batch_size):
            # Input and output data generation
            image = image_generator(image_size)
            message = message_generator(message_length, dictionary_length)
            message_onehot = onehot.onehot_encoder(message,
                                                   dictionary_length)

            input_images[i] = image
            input_messages[i] = message
            output_images[i] = image
            output_messages[i] = message_onehot

        yield [[input_images, input_messages],
               [output_images, output_messages]]
