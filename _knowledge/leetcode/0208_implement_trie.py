class Trie:

    def __init__(self):
        self.set_words = set()
        self.set_prefixes = set()

    def insert(self, word: str) -> None:
        self.set_words.update({word})
        list_new_prefixes = [word[:lenght] for lenght in range(1, len(word) + 1)]
        self.set_prefixes.update(set(list_new_prefixes))

    def search(self, word: str) -> bool:
        return word in self.set_words

    def startsWith(self, prefix: str) -> bool:
        return prefix in self.set_prefixes


trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    # return True
print(trie.search("app"))      # return False
print(trie.startsWith("app"))  # return True
trie.insert("app")
print(trie.search("app"))      #return True
