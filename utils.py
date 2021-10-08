class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print_model(model):
    """ Print the model as LaTeX tabular """
    print("\n")
    print(r"\begin{tabular}{l|l|l}")
    print(r"     layer & shape & activation \\")
    print(r"     \hline")
    # go through each layer
    for layer in model.layers:
        # name
        name = layer.__class__.__name__.replace(r"_", r"\_")
        print(f"    {name} ", end="")
        # shape
        print(r"& $", end="")
        shape = list(layer.output_shape)
        if layer.name == "input_1": shape = shape[0]
        shape = shape[1:]
        for i in range(len(shape)):
        print(f"{shape[i]}", end="")
        if i < len(shape)-1: print(r" \times ", end="")
        print(r"$ ", end="")
        #  activation function
        config = layer.get_config()
        if "activation" in config.keys():
            print(f"& {config['activation']} ", end="")
        else:
            print(r"& ", end="")
        print(r"\\")
    print(r"\end{tabular}")
    print("\n")
