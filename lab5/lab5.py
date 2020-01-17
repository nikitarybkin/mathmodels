import itertools
import re
from collections import Counter
import matplotlib.pyplot as plt
# from progressbar2 import *
import progressbar

import numpy as np

source_str = open('karenina.txt', 'r').read()
source_words = re.findall('(\w+)|\n\n+|[.?!]', source_str, re.UNICODE)
# print('Список слов:',source_words[:10])
# print('Всего слов, включая концы предложений:',len(source_words))

all_words = Counter(source_words)
number_of_sentences = all_words.pop('', None)
print('Всего предложений:',number_of_sentences)
print('Всего различных слов:',len(all_words))
print('Самые часто встречающиеся слова:',all_words.most_common(10))


sentences = [list(y) for x, y in itertools.groupby(source_words, lambda z: z == '') if not x]
for sentence in sentences:
    first_word_lower_case = sentence[0].lower()
    if first_word_lower_case in all_words: sentence[0] = first_word_lower_case
print('Начало текста:',sentences[:6])

all_words = Counter([word for sentence in sentences for word in sentence])
number_of_sentences = len(sentences)

least_count = 50
max_count = 10000
common_words = {k: v for k, v in all_words.items() if v >= least_count and v <= max_count}

words_sorted_by_frequency = list(map(lambda x: x[0], sorted(common_words.items(), key=lambda x: -x[1])))
words_codes = dict([(w, c) for c, w in enumerate(words_sorted_by_frequency)])
encoded_text = [[words_codes[word] for word in sentence if word in words_codes] for sentence in sentences]
minimal_length = 4
encoded_text = [sentence for sentence in encoded_text if len(sentence) >= minimal_length]




def decode(txt):
    return words_sorted_by_frequency[txt] if isinstance(txt, (int, np.integer)) else list(map(decode, list(txt)))


print('Раскодированное начало:', decode(encoded_text[:6]))


def random_context(text, distance_between_words=5):
    random_sentence = text[np.random.randint(0, len(text))]
    word_index = np.random.randint(0, len(random_sentence))
    word = random_sentence[word_index]
    left_context = random_sentence[max(0, word_index - distance_between_words):word_index]
    right_context = random_sentence[(word_index + 1):min(len(random_sentence), word_index + distance_between_words)]
    return word, left_context + right_context



def loss_function_sq(Dx, y):
    delta = Dx - y
    return np.dot(delta, delta), delta


def loss_function_cr(tx, y):
    return -np.sum(tx * np.log(y)), -tx / y


def linear_layer(theta, x):
    if np.shape(x) == (559,):
        i, = np.where(x == 1)
        return theta[:, i[0]]
    return np.dot(theta, x)


def linear_layer_dx(dx, theta):
    return np.dot(dx, theta)


def linear_layer_dtheta(dx, x):
    return dx[:, None] * x[None, :]


def logistic_function(x):
    t = np.exp(-x)
    return 1 / (1 + t), t


def logistic_function_dx(dx, y, t):
    return dx * t * y * y


def softmax(x):
    t = np.exp(-x)
    return t / np.sum(t)


def softmax_dx(dx, y):
    # y=softmax(x)
    w = y * dx
    return y * np.sum(w) - w


dimensionality = len(words_sorted_by_frequency)


def context_to_vector(context):
    result = np.zeros(dimensionality)
    for word in context: result[word] += 1. / len(context)
    # print(result)
    return result


print("Число слов в случайном контексте:", np.sum(0 != context_to_vector(random_context(encoded_text)[1])))


def neural_network(A, B, x):
    return softmax(linear_layer(B, softmax(linear_layer(A, x))))[0]


def loss(A, B, x, tx, return_grad=True):
    z3 = linear_layer(A, x)
    z2 = linear_layer(B, z3)
    z1 = softmax(z2)
    R, dz1 = loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2 = softmax_dx(dz1, z1)
    dB = linear_layer_dtheta(dz2, z3)
    dz3 = linear_layer_dx(dz2, B)
    dA = linear_layer_dtheta(dz3, x)
    return R, dA, dB


def grad(A, B, x, tx, return_grad=True):
    z3 = linear_layer(A, x)
    z2 = linear_layer(B, z3)
    z1 = softmax(z2)
    R, dz1 = loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2 = softmax_dx(dz1, z1)
    dz3 = linear_layer_dx(dz2, B)
    dx = linear_layer_dx(dz3, A)
    return R, dx

def make_batch(text, size=300):
    C = [];
    W = []
    for _ in range(size):
        word, context = random_context(text)
        W.append(context_to_vector([word]))
        C.append(context_to_vector(context))
    return W, C


def train_on_batch(A, B, batch, number_of_steps=3, step_size=5):
    batch_size = len(batch[0])
    history = []
    for _ in range(number_of_steps):
        dA = np.zeros(A.shape);
        dB = np.zeros(B.shape);
        error = 0
        for W, C in zip(*batch):
            R, DA, DB = loss(A, B, W, C)
            error += R;
            dA += DA;
            dB += DB
        A -= step_size / batch_size * dA
        B -= step_size / batch_size * dB
        history.append(error / batch_size)
    return history


def train_network(A, B, text, test=None, number_of_steps=100, debug=False):
    report_each = number_of_steps / 10
    history = []
    if not test is None:
        print("Initial error {}".format(test_network(A, B, test_text)))
    try:
        for n in progressbar.progressbar(range(1, number_of_steps + 1), name='Batch'):
            error = train_on_batch(A, B, make_batch(text))
            if debug: print(n, ":", error[-1])
            if not test is None and n % report_each == 0:
                print("Epoch {}, generalization error {}".format(n, test_network(A, B, test_text)))
            history.extend(error)
    except KeyboardInterrupt:
        pass
    return history


def test_network(A, B, text, number_of_samples=100):
    error = 0
    for _ in range(number_of_samples):
        word, context = random_context(text)
        W = context_to_vector([word])
        C = context_to_vector(context)
        error += loss(A, B, W, C, return_grad=False)
    return error / number_of_samples


ratio = 0.9
train_text = encoded_text[:int(ratio * len(encoded_text))]
test_text = encoded_text[int(ratio * len(encoded_text)):]
features = 50
A = np.random.rand(features, dimensionality)
B = np.random.rand(dimensionality, features)

history = train_network(A, B, train_text, test=test_text)
plt.plot(history, ',')
plt.xlabel("Training step")
plt.ylabel("Error")
plt.show()


def word2vec(B, n):
    return B[n, :]


def similar_words(B, vec):
    dist = np.sum((B - vec[None, :]) ** 2, axis=1)
    return sorted(enumerate(dist), key=lambda x: x[1])


def distance_matrix(B):
    return np.sum((B[:, :, None] - B[:, None, :]) ** 2, axis=0)


print(words_sorted_by_frequency)


def show_similar(B, vec, count=10):
    for code, freq in similar_words(B, vec)[:count]:
        print("{}/{}".format(decode(code), int(freq)), end=" ")


words_to_compare = ['Анна', 'Степан', 'давно', 'много', 'руки']
for word in words_to_compare:
    print(word, ":", end=" ")
    show_similar(B, word2vec(B, words_codes[word]))
    print()


# Попробуем проанализировать отношения слов
def show_relative(n1, n2, n3):
    w1 = words_codes[n1];
    v1 = word2vec(B, w1)
    w2 = words_codes[n2];
    v2 = word2vec(B, w2)
    w3 = words_codes[n3];
    v3 = word2vec(B, w3)
    v4 = v2 - v1 + v3
    print("Как '{}' относится к '{}', также к '{}' относятся следующие слова в порядке убывания уверенности:".format(
        decode(w1), decode(w2), decode(w3)))
    show_similar(B, v4)
    print("\n")


show_relative('Анна', 'Степан', 'она')
show_relative('Степан', 'Анна', 'он')

# Выведем все вектора слов
plt.imshow(np.abs(B.T), interpolation='none')
plt.show()

# Выведем матрицу расстояний между словами
plt.imshow(distance_matrix(B), interpolation='none')
plt.show()

n1 = 'он';
n2 = 'она'
ns = ['Анна', 'Степан', 'жена', 'хорошо', 'руки']
w1 = words_codes[n1];
w2 = words_codes[n2];
v1 = word2vec(B, w1);
v2 = word2vec(B, w2);
p1 = np.dot(B, v1);
p2 = np.dot(B, v2);
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(p1, p2, '.')
for n in ns:
    w = words_codes[n]
    ax.plot(p1[w], p2[w], '.r')
    ax.annotate(n, xy=(p1[w], p2[w]), color='r')
plt.xlabel(n1);
plt.ylabel(n2);
plt.show()
