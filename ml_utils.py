import pickle

def save_scikit_like_model_params(fitted_model, path):
    params = fitted_model.get_params()
    with open(path, "wb") as file:
        pickle.dump(params, file)


def load_scikit_like_model_params(model, path):
    with open(path, "rb") as file:
        params = pickle.load(file)
        model.set_params(params)
        return model