class GSN1RandomSearchFilter:
    def __init__(self):
        pass

    def __call__(self, variation):
        # Filter based on your original rule
        if variation["model.n_conv_layers"] >= 9 and variation['model.maxpool_placing'] == 'even_convs':
            return False

        # Heuristic rule: filter out high learning rate with small batch size
        lr = variation['optimizer.lr']
        batch_size = variation['trainer.batch_size']

        # Rule 1: For batch size 8 or 16, filter out if lr > 0.00001
        if batch_size in [8, 16] and lr > 0.00001:
            return False

        # Rule 2: For batch size 64 or 128, filter out if lr < 0.00001
        if batch_size in [64, 128] and lr < 0.00001:
            return False

        # If no rule applies, keep the variation
        return True
