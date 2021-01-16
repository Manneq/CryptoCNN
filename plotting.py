"""
    File 'plotting.py' with functions to plot Neural Network.
"""
import keras


def model_plotting(model, path_to_file):
    """
        Method to plot models.
        param:
            1. model - keras Neural Network model
            2. path_to_file - string path to save file
    """
    keras.utils.plot_model(model,
                           to_file=path_to_file,
                           show_shapes=True)

    return
