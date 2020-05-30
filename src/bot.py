#!/usr/bin/env python3

import os
import random
from dotenv import load_dotenv
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

load_dotenv()

PROXY = os.getenv('PROXY')
TOKEN = os.getenv('TOKEN')
CLASSIFIER_TRESHOLD = 0.2
GENERATIVE_TRESHOLD = 0.6
SCRIPT_DATA = {
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
        'Извините, я вас не понял',
        'Простите, вы о чем?',
    ]
}

# Глобальные переменные
stats = {
    'requests': 0,
    'byscript': 0,
    'bygenerative': 0,
    'stub': 0
}
GENERATIVE_DIALOGUES = []

# Заполнение GENERATIVE_DIALOGUES
with open('dialogues.txt') as f:
    data = f.read()
dialogues = []
for dialogue in data.split('\n\n'):
    replicas = []
    for replica in dialogue.split('\n')[:2]:
        replica = replica[2:].lower()
        replicas.append(replica)
    if len(replicas) == 2 and len(replicas[0]) > 1 and len(replicas[1]) > 0:
        dialogues.append(replicas)
GENERATIVE_DIALOGUES = dialogues[:5000]

# Глобальные переменные и обучение классификатора
X_text = []
y = []
for intent, value in SCRIPT_DATA['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)
VECTORIZER = CountVectorizer()
X = VECTORIZER.fit_transform(X_text)
CLF = LogisticRegression()
CLF.fit(X, y)


def get_intent(text):
    probas = CLF.predict_proba(VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if max_proba >= CLASSIFIER_TRESHOLD:
        index = list(probas[0]).index(max_proba)
        return CLF.classes_[index]


def get_answer_by_generative_model(text):
    text = text.lower()

    for question, answer in GENERATIVE_DIALOGUES:
        if abs(len(text) - len(question)) / len(question) < 1 - GENERATIVE_TRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            if similarity > GENERATIVE_TRESHOLD:
                return answer


def get_response_by_intent(intent):
    responses = SCRIPT_DATA['intents'][intent]['responses']
    return random.choice(responses)


def get_failure_phrase():
    phrases = SCRIPT_DATA['failure_phrases']
    return random.choice(phrases)


def generate_answer(text):
    stats['requests'] += 1

    # Make answer by script
    intent = get_intent(text)
    if intent:
        stats['byscript'] += 1
        response = get_response_by_intent(intent)
        return response

    # Or use generative model
    answer = get_answer_by_generative_model(text)
    if answer:
        stats['bygenerative'] += 1
        return answer

    # Or use one of failure answers
    stats['stub'] += 1
    failure_phrase = get_failure_phrase()
    return failure_phrase


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def text(update, context):
    """Ответ уже от движка"""
    answer = generate_answer(update.message.text)
    print(update.message.text, '->', answer)
    print(stats)
    print()
    update.message.reply_text(answer)


def error(update, context):
    update.message.reply_text('Я работаю только с текстом')


def main():
    updater = Updater(TOKEN, request_kwargs={
                      'proxy_url': PROXY}, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, text))
    dp.add_error_handler(error)
    print('hello')
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
