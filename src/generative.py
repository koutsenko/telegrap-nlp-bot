import sys
from nltk.metrics.distance import edit_distance


this = sys.modules[__name__]


GENERATIVE_TRESHOLD = 0.3


def fill_dialogues():
    with open('dialogues.txt', encoding='utf-8') as f:
        data = f.read()
    dialogues = []
    for dialogue in data.split('\n\n'):
        replicas = []
        for replica in dialogue.split('\n')[:2]:
            replica = replica[2:].lower()
            replicas.append(replica)
        if len(replicas) == 2 and len(replicas[0]) > 1 and len(replicas[1]) > 0:
            dialogues.append(replicas)
    this.GENERATIVE_DIALOGUES = dialogues[:5000]


def get_answer_by_generative_model(text):
    text = text.lower()
    for question, answer in this.GENERATIVE_DIALOGUES:
        if abs(len(text) - len(question)) / len(question) < 1 - this.GENERATIVE_TRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            if similarity > this.GENERATIVE_TRESHOLD:
                return answer
