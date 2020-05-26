#!/usr/bin/env python
# coding: utf-8

# Обучение классификатора

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# TODO Заменить на полную версию
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

# Будущий список словаря с фразами
X_text = [] 

# Соответствующие этим фразам классы (интенты)
y = []

# Читаем конфиг бота и заполняем список фраз и соответствующих им классов
for intent, value in BOT_CONFIG['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)

# Создадим векторайзер, для превращения строк текста в вектора с кол-вом вхождений слов 
vectorizer = CountVectorizer()

# Обучаем векторайзер, сохраняем его базу знаний в X 
X = vectorizer.fit_transform(X_text)

# Векторайзер: список фич, т.е. отдельных слов, со вхождениями которых мы работаем
# print(vectorizer.get_feature_names())

# Векторайзер: список сохраненных векторов
# print(X.toarray())

# Новые фразы в векторном виде с точки зрения нашего векторайзера будут выглядеть так:
# vectorizer.transform(["привет, бот 343434", "пока бот", "привет как дела"]).toarray()

# Классификатор на базе линейной модели "логистическая регрессия"
clf = LogisticRegression()

# Код ниже сгенерит ошибку - модель еще не обучена.
# print(clf.classes_)

# Обучение классификатора
clf.fit(X, y)

# Известные теперь классы
print("Known classes:", clf.classes_)

# Тестовые фразы в векторном виде
test = ["привет, бот 343434", "пока бот", "привет как дела"]
print("Test phrases: ", test)

# Тестовые фразы в векторном виде
test_vectorized = vectorizer.transform(test)

# Распознаем на обученном классификаторе нижеприведенные фразы
print("Recognized:", clf.predict(test_vectorized))

# Вероятности по каждой фразе, например, в 1-й фразе это вторая ячейка, т.е. второй класс - "hello"
print("Probabilities:", clf.predict_proba(test_vectorized))



