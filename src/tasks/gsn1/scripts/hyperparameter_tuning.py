"""
func_module = "src.tasks.gsn1.scripts.hyperparameter_tuning"
module = importlib.import_module(func_module)
func = getattr(module, cfg.func_name)
result = func(variation)
"""
def random_search_filter(variation):
    _v = variation
    if _v["model.n_conv_layers"] >= 9 and _v['model.maxpool_placing'] == 'even_convs':
        return False
    else:
        return True

