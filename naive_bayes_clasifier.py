import numpy as np
import math
import statistics as st


class possible_class:
    def __init__(self, name):
        self.name = name
        self.params = []
        self.counter = 0
        self.probability = 0


class Bayes_clasifier:
    def __init__(self):
        self.poss_classes = []

    def find_class(self, name):
        for one in self.poss_classes:
            if one.name == name:
                return one

    def train(self, x_val, y_val):
        used_classes = []

        for num, one in enumerate(x_val):
            for ind, param in enumerate(one):
                if y_val[num] not in used_classes:
                    used_classes.append(y_val[num])
                    c1 = possible_class(y_val[num])
                    self.poss_classes.append(c1)
                class_obj = self.find_class(y_val[num])
                if ind >= len(class_obj.params):
                    p1 = Parameter()
                    class_obj.params.append(p1)
                class_obj.params[ind].add_data(param)
            class_obj.counter += 1
        for one in self.poss_classes:
            for param in one.params:
                param.find_mean()
                param.find_sigma()
        self.set_class_probabilities(len(x_val))

    def set_class_probabilities(self, num_of_el):
        for one in self.poss_classes:
            one.probability = one.counter / num_of_el

    def probability_counter(self, value, param):
        mean = param.mean
        sigma = param.sigma
        result = 1 / (math.sqrt(2 * sigma**2 * math.pi))
        eval = math.exp((-((value - mean) ** 2)) / (2 * sigma**2))
        return result * eval

    def posterior(self, class1, params_values):
        posterior_val = class1.probability
        for In, param in enumerate(class1.params):
            posterior_val *= self.probability_counter(params_values[In], param)
        return posterior_val

    def predict(self, x_val):
        posteriors_vals = []
        for class1 in self.poss_classes:
            posteriors_vals.append(self.posterior(class1, x_val))
        return self.poss_classes[self.find_max(posteriors_vals)].name

    def find_max(self, vector):
        max_val = vector[0]
        max_ind = 0
        for ind, one in enumerate(vector):
            if one > max_val:
                max_val = one
                max_ind = ind
        return max_ind


class Parameter:
    def __init__(self):
        self.values = []

    def add_data(self, data):
        self.values.append(data)

    def find_mean(self):
        self.mean = st.mean(self.values)

    def find_sigma(self):
        self.sigma = np.std(self.values)
