#!/usr/bin/env python
# coding: utf-8

import random
from nltk.metrics.distance import edit_distance


BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Здравствуйте', 'Шалом'],
            'responses': ['Привет, юзер!', 'Здравствуй']
        },
        'goodbye': {
            'examples': ['Пока', 'Всего доброго', 'До свидания'],
            'responses': ['Пока', 'Счастливо!']
        },
        'thanks': {
            'examples': ['Спасибо', 'Спасибо большое!', 'Сенкс', 'Благодарю'],
            'responses': ['Вам спасибо!']
        },
        'whatcanyoudo': {
            'examples': ['Что ты умеешь?', 'Расскажи что умеешь'],
            'responses': ['Отвечать на вопросы. Просто напиши :)']
        },
        'name': {
            'examples': ['Как тебя зовут?', 'Твое имя?'],
            'responses': ['Меня зовут бот. Просто бот.']
        },
        'weather': {
            'examples': ['Какая погода в Москве?', 'Какая погода?'],
            'responses': ['Погода так себе...']
        }
    },
    'failure_phrases': [
        'Я не знаю, что ответить',
        'Не поняла вас',
        'Простите, не поняла вас',
        'Я на такое не умею отвечать',
        'Переформулируйте, пожалуйста'
    ]
}

def get_intent(text):
    intents = BOT_CONFIG['intents']

    for intent, value in intents.items():
        for example in value['examples']:
            dist = edit_distance(text, example)          
            difference = dist / len(example)
            similarity = 1 - difference
            if similarity > 0.6:
                return intent


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


def generate_answer(text):
    # NLU
    intent = get_intent(text)
    
    # Make answer
    
    # by script    
    if intent:
        response = get_response_by_intent(intent)
        return response
    
    # use generative model
    # TODO ...
    
    # use stub
    failure_phrase = get_failure_phrase()
    return failure_phrase    


while True:
    text = input('Введите вопрос: ')
    answer = generate_answer(text)
    print(answer)
