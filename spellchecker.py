import textdistance
import numpy as np

import config


distance_functions = [
    textdistance.damerau_levenshtein.normalized_distance,
    textdistance.needleman_wunsch.normalized_distance,
    textdistance.editex.normalized_distance,
]


class SpellChecker:
    def __init__(self, dictionary_path=config.dictionary_path):
        with open(dictionary_path, 'r') as f:
            self.dictionary = np.array([line[:-1].split('/')[0] for line in f.readlines()])
        self.finder = WordFinder()

    def check(self, word, top=5):
        if word in self.dictionary:
            print('The word is OK.')
        else:
            print('The word is incorrect!')
            print('Trying to fix...')
            similar = self.finder.get_similar(word, top)
            print('Did you mean one of these: ', end='')
            for sim in similar:
                print(sim, end=', ')
            print('\b\b?')


class WordFinder:
    def __init__(self, dictionary_path=config.dictionary_path, distance_functions=distance_functions):
        with open(dictionary_path, 'r') as f:
            self.dictionary = np.array([line[:-1].split('/')[0] for line in f.readlines()])
        self.distance_functions = distance_functions

    def get_similar(self, word, top=5):
        distances = np.zeros(len(self.dictionary))
        for i, dict_word in enumerate(self.dictionary):
            for func in self.distance_functions:
                distances[i] += func(word, dict_word)
        top_idx = np.argpartition(distances, top)[:top]
        return self.dictionary[top_idx]