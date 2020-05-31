# Модуль для ввода\вывода через телеграм. Висит, слушает сообщения, выдает ответы.
# Функция выдачи ответов передается как параметр из родительского модуля.
# Вопрос правильной организации взаимодействия остается открытым.
# Равно как и необходимость переписывания на классы.

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from dotenv import load_dotenv

import os

load_dotenv()

PROXY = os.getenv('PROXY')
TOKEN = os.getenv('TOKEN')


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def error(update, context):
    update.message.reply_text('Я работаю только с текстом')


def text(update, context, generate_answer, print_stats):
    """Ответ уже от движка"""
    answer = generate_answer(update.message.text)
    update.message.reply_text(answer)
    print(update.message.text, '->', answer)
    print_stats()


def run(generate_answer, print_stats):

    def curried_text(update, context):
        return text(update, context, generate_answer, print_stats)

    updater = Updater(TOKEN, request_kwargs={
                      'proxy_url': PROXY}, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, curried_text))
    dp.add_error_handler(error)
    print('hello')
    updater.start_polling()
    updater.idle()
