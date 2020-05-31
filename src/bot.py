#!/usr/bin/env python3

import random

import classifier
import failure
import generative
import transport_telegram


stats = {
    'requests': 0,
    'byscript': 0,
    'bygenerative': 0,
    'stub': 0
}


def generate_answer(text):
    stats['requests'] += 1

    intent = classifier.get_intent(text)
    if intent:
        stats['byscript'] += 1
        response = classifier.get_response_by_intent(intent)
        return response

    answer = generative.get_answer_by_generative_model(text)
    if answer:
        stats['bygenerative'] += 1
        return answer

    stats['stub'] += 1
    failure_phrase = failure.get_failure_phrase()
    return failure_phrase


def print_stats():
    print(stats)


def main():
    classifier.study()
    generative.fill_dialogues()
    transport_telegram.run(generate_answer, print_stats)


if __name__ == '__main__':
    main()
