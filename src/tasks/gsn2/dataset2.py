import pickle

def get_mnist_data():
    with open('../../../data/gsn2/mnist_data.pkl', 'rb') as f:
        mnist_data = pickle.load(f)
        return mnist_data

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = get_mnist_data()
for x in mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test:
    print(type(x), x.shape)