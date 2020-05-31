import random
import sys


this = sys.modules[__name__]


FAILURE_PHRASES = [
    'Извините, я вас не понял',
    'Простите, вы о чем?',
]


def get_failure_phrase():
    return random.choice(this.FAILURE_PHRASES)
