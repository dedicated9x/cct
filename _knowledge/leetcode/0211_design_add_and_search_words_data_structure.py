class WordDictionary:

    def __init__(self):
        self.set_words = set()

    def addWord(self, word: str) -> None:
        word_as_list = list(word)

        list_combinations = []
        list_combinations.append(word)

        for i in range(len(word)):
            for j in range(len(word)):
                base_word = word_as_list.copy()
                base_word[i] = "."
                base_word[j] = "."
                new_word = "".join(base_word)
                list_combinations.append(new_word)

        self.set_words.update(set(list_combinations))


    def search(self, word: str) -> bool:
        return word in self.set_words


wordDictionary = WordDictionary()
wordDictionary.addWord("bad")
wordDictionary.addWord("dad")
wordDictionary.addWord("mad")
print(wordDictionary.search("pad")) # return False
print(wordDictionary.search("bad")) # return True
print(wordDictionary.search(".ad")) # return True
print(wordDictionary.search("b..")) # return True