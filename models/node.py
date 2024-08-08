class Node:
    def __init__(self, key, value, type_):

        self.key = key
        self.value = value
        self.type_ = type_
        self.children = []

    def __str__(self):
        return f'<{self.type_} NODE\t{self.key}\tCHILDREN=[{len(self.children)}]\tVALLUE={self.value}]>'