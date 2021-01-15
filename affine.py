"""
    File 'affine.py' consists of methods for Affine cipher.
"""


def egcd(a, b):
    """
       Extended Euclidean Algorithm for modular inverse
        param:
            1. a - first key for Affine cipher
            2. b - length of the dictionary
        return:
            Values for modular inverse.
    """
    x, y, u, v = 0, 1, 1, 0

    while a != 0:
        q, r = b // a, b % a
        m, n = x - u * q, y - v * q
        b, a, x, y, u, v = a, r, u, v, m, n

    return b, x, y


def modular_inverse(key_1, dictionary_length):
    """
        Method to find modular inverse for Affine cipher
        param:
            1. key_1 - first key for Affine cipher
            2. dictionary_length - length of the dictionary
        return:
            Modular inverse
    """
    gcd, x, y = egcd(key_1, dictionary_length)

    if gcd != 1:
        raise Exception("Modular inverse does not exist!")
    else:
        return x % dictionary_length


def affine_encryption(message, key_1, key_2, dictionary_length):
    """
        Method to encrypt message using Affine cipher.
        param:
            1. message - message that need to be encrypted
            2. key_1 - first key for Affine cipher
            3. key_2 - second key for Affine cipher
            4. dictionary_length - length of the dictionary
        return:
            Message encrypted using Affine cipher
    """
    return ''.join(
        [chr(((key_1 * symbol.encode("ascii")[0] + key_2) % dictionary_length))
         for symbol in message])


def affine_decryption(message, key_1, key_2, dictionary_length):
    """
        Method to decrypt message using Affine cipher.
        param:
            1. message - message that need to be decrypted
            2. key_1 - first key for Affine cipher
            3. key_2 - second key for Affine cipher
            4. dictionary_length - length of the dictionary
        return:
            Message encrypted using Affine cipher
    """
    return ''.join([chr(
        ((modular_inverse(key_1, dictionary_length) *
          (symbol.encode("ascii")[0] - key_2)) % dictionary_length))
        for symbol in message])
