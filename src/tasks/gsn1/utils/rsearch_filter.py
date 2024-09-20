class GSN1RandomSearchFilter:
    def __init__(self):
        pass

    def __call__(self, variation):
        if variation["model.n_conv_layers"] >= 9 and variation['model.maxpool_placing'] == 'even_convs':
            return False
        else:
            return True
