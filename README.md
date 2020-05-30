# telegrap-nlp-bot

Шаги для запуска<br/>
1. Установить зависимости<br/>
```$ pip install -r requirements.txt```

2. Скопировать .env.example в .env:<br/>
```$ cp .env.example .env```

3. Прочесть комментарии и заполнить .env, нужны прокси и токен:<br/>
```$ cat .env```

4. Распаковать словарь (взят [отсюда](https://github.com/Koziev/NLP_Datasets/blob/master/Conversations/Data/dialogues.zip)):<br/>
```$ cd src && unzip dialogues.zip```

5. Запустить. Выведет hello - всё, можно писать ему в телеге в лс:<br/>
```$ python bot.py```
