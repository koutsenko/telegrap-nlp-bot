import random
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


this = sys.modules[__name__]


CLASSIFIER_TRESHOLD = 0.2


INTENTS = {
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
}


CLF = LogisticRegression()
VECTORIZER = CountVectorizer()


def study():
    X_text = []
    y = []
    for intent, value in this.INTENTS.items():
        for example in value['examples']:
            X_text.append(example)
            y.append(intent)
    X = this.VECTORIZER.fit_transform(X_text)
    this.CLF.fit(X, y)


def get_intent(text):
    probas = this.CLF.predict_proba(this.VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if max_proba >= this.CLASSIFIER_TRESHOLD:
        index = list(probas[0]).index(max_proba)
        return this.CLF.classes_[index]


def get_response_by_intent(intent):
    responses = this.INTENTS[intent]['responses']
    return random.choice(responses)
