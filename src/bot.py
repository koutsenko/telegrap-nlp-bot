#!/usr/bin/env python
# coding: utf-8

import random
from nltk.metrics.distance import edit_distance


BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Йо', 'Хай', 'Дарова', 'Привет', 'Добрый день', 'Здравствуй', 'Шалом'],
            'responses': ['Йо', 'Хай', 'Дарова', 'Привет', 'Добрый день', 'Здравствуй', 'Шалом']
        },
        'goodbye': {
            'examples': ['Счастливо', 'Пока', 'Всего доброго', 'До свидания', 'До встречи', 'Давай', 'Бывай'],
            'responses': ['Счастливо', 'Пока', 'Всего доброго', 'До свидания', 'До встречи', 'Давай', 'Бывай'],
        },
        'thanks': {
            'examples': ['Спасибо', 'Спасибо большое', 'Сенкс', 'Благодарю', 'Мерси', 'Спс'],
            'responses': ['Вам спасибо!', 'Не за что', 'Пожалуйста', 'На здоровье', 'Обращайтесь']
        },
        'whatcanyoudo': {
            'examples': ['Что ты умеешь?', 'Расскажи что умеешь', 'Помощь', 'Справка', 'Возможности'],
            'responses': ['Я могу отвечать на вопросы. Просто напиши :)']
        },
        'name': {
            'examples': ['Как тебя зовут?', 'Твое имя?', 'Как к тебе обращаться?', 'Как тебя назвали?', 'Имя', 'Как зовут'],
            'responses': ['Меня зовут бот. Просто бот.']
        },
        'weather': {
            'examples': ['Какая погода в Москве?', 'Какая погода?', 'Прогноз погоды?', 'Что с погодой', 'Как на улице?'],
            'responses': ['Погода так себе...']
        },
        'mood': {
            'examples': ['Как себя чувствуешь', 'Как настроение', 'Ты как', 'Как ты', 'Как ты там', 'Все нормально?', 'Как дела'],
            'responses': ['Нормально', 'Не очень', 'Отлично', 'Бывает и лучше', 'Нормуль', 'Всё ОК']
        },
        'advice': {
            'examples': ['Дай совет', 'Нужен совет', 'Как поступить', 'Что мне делать', 'Что делать', 'Посоветуй', 'Подскажи', 'Как думаешь'],
            'responses': ['Не бойся принимать решение', 'Осознай сомнения и проработай', 'Не тормози', 'Нормально делай - нормально будет']
        },
        'wishes': {
            'examples': ['Доброе утро', 'Хорошего дня', 'Как поступить', 'Что мне делать', 'Что делать', 'Посоветуй', 'Подскажи', 'Как думаешь'],
            'responses': ['Спасибо', 'Взаимно', 'И тебе']
        },
        'ask': {
            'examples': ['Почему', 'А чего так', 'Как так', 'Зачем'],
            'responses': ['Если б знал', 'Не знаю', 'Да фиг его знает', 'Разное бывает']
        },
        'complain': {
            'examples': ['Я устал', 'Устал', 'Мне плохо', 'Заколебался', 'Все достало', 'Нет сил', 'Ну и денек', 'Все плохо'],
            'responses': ['Возьми себя в руки', 'Бывает, не парься', 'Сделай паузу и глубоко вдохни', 'Не принимай близко к сердцу', 'Не бери в голову']
        },
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
