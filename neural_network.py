"""
    File 'neural_network.py' consists of methods to create.
"""
import keras
import numpy as np
import data_generator
import data_management
import onehot


def model_creation(image_size, message_length, dictionary_length):
    """
        Method to create model.
        param:
            1. image_size - size of the image
            2. message_length - maximum length of the message
            3. dictionary_length - size of the dictionary
        return:
            1. model - Keras Neural Network model
            2. encoder - encoder part of the model
            3. decoder - decoder part of the model
    """
    # Encoder model
    image_input = keras.layers.Input(image_size, name="image_input")
    sentence_input = keras.layers.Input((message_length,),
                                        name="sentence_input")
    sentence_embedding = \
        keras.layers.Embedding(dictionary_length, 100)(sentence_input)
    sentence_embedding = keras.layers.Flatten()(sentence_embedding)
    sentence_embedding = \
        keras.layers.Reshape((image_size[0], image_size[1], 1))(
            sentence_embedding)
    image_convolutional = \
        keras.layers.Conv2D(20, 1, activation='tanh')(image_input)
    concatenation = keras.layers.Concatenate(axis=-1)(
        [sentence_embedding, image_convolutional])
    image_output = keras.layers.Conv2D(3, 1, activation='tanh')(concatenation)
    encoder_model = keras.models.Model(inputs=[image_input, sentence_input],
                                       outputs=[image_output], name="encoder")

    # Decoder model
    decoder_model = keras.models.Sequential(name="decoder")
    decoder_model.add(keras.layers.Conv2D(1, 1, input_shape=(100, 100, 3)))
    decoder_model.add(keras.layers.Reshape((message_length, 100)))
    decoder_model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(dictionary_length, activation='softmax')))
    sentence_output = decoder_model(image_output)

    # Model creation
    model = keras.models.Model(inputs=[image_input, sentence_input],
                               outputs=[image_output, sentence_output])
    model.compile('Adadelta',
                  loss=['mean_absolute_error', 'categorical_crossentropy'],
                  metrics={'decoder': 'categorical_accuracy'})

    return model, encoder_model, decoder_model


def model_training(image_size, message_length, dictionary_length):
    """
        Method to train model.
        param:
            1. image_size - size of the image
            2. message_length - maximum length of the message
            3. dictionary_length - size of the dictionary
    """
    # Training generator initializing
    training_data_generator = \
        data_generator.training_data_generator(image_size, message_length,
                                               dictionary_length)

    # Model creation
    model, encoder, decoder = \
        model_creation(image_size, message_length, dictionary_length)

    # Model training
    model.fit_generator(training_data_generator,
                        epochs=512,
                        steps_per_epoch=384,
                        callbacks=[
                            keras.callbacks.TerminateOnNaN(),
                            keras.callbacks.ReduceLROnPlateau(
                                monitor='loss',
                                patience=5,
                                verbose=1),
                            keras.callbacks.EarlyStopping(
                                monitor='loss',
                                patience=11,
                                verbose=1),
                            keras.callbacks.ModelCheckpoint(
                                "weights.h5",
                                monitor='loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True),
                            keras.callbacks.TensorBoard(
                                log_dir="training_logs")])

    return


def model_evaluation(path_to_image_folder, path_to_words_set,
                     image_size, message_length, dictionary_length):
    """
        Method to evaluate model.
        param:
            1. path_to_image_folder - string path to image folder
            2. path_to_words_set - string path to the message file
            3. image_size - size of the image
            4. message_length - maximum length of the message
            5. dictionary_length - size of the dictionary
    """
    # Data preparation
    images = data_management.images_names_loading(path_to_image_folder,
                                                  message_length)
    input_messages = \
        data_management.messages_from_text_loading(path_to_words_set,
                                                   message_length,
                                                   len(images))

    output_messages = np.zeros((len(input_messages), message_length,
                                dictionary_length))

    for i in range(len(input_messages)):
        output_messages[i] = onehot.onehot_encoder(input_messages[i],
                                                   dictionary_length)

    # Model creation
    model, encoder, decoder = \
        model_creation(image_size, message_length, dictionary_length)

    # Model evaluation
    metrics = model.evaluate(x=[images, input_messages],
                             y=[images, output_messages],
                             batch_size=64)

    print(model.metrics_names)
    print(metrics)

    return


def model_loading(image_size, message_length, dictionary_length):
    """
        Method to load model and weights.
        param:
            1. image_size - size of the image
            2. message_length - maximum length of the message
            3. dictionary_length - size of the dictionary
        return:
            1. encoder - encoder part of the model
            2. decoder - decoder part of the model
    """
    # Model creation
    model, encoder, decoder = \
        model_creation(image_size, message_length, dictionary_length)

    # Weights loading
    model.load_weights("weights.h5")

    return model, encoder, decoder


def model_encoding(encoder, image, message):
    """
        Method to encode message.
        param:
            1. encoder - encoder part of the model
            2. image - image to encode
            3. message - ascii message to encode
        return:
            Encoded image
    """
    return encoder.predict([image, message])[0]


def model_decoding(decoder, image):
    """
        Method to decode message.
        param:
            1. decoder - decoder part of the model
            2. image - encoded image
        return:
            Decoded onehot message
    """
    return decoder.predict(image)[0]


def model_test(encoder, decoder, image, message):
    """
        Method to encrypt and decrypt message in one-shot.
        param:
            1. encoder - encoder part of the model
            2. decoder - decoder part of the model
            3. image - image to encode
            4. message - message in ascii codes to encode
        return:
            1. Encoded image
            2. Decoded message
    """
    # Encode
    image = encoder.predict([image, message])

    # Decode
    message = decoder.predict(image)[0]

    return image[0], message
