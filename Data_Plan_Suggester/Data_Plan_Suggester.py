import keras.utils
import numpy as np
from os import path
from keras import callbacks
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation

def train(name, plans_mb, plans_template, standard_deviations, n_each, epochs):

    input_size = 1
    batch_size = 100
    hidden_neurons = 100
    usage_temp = np.zeros((len(plans_template), n_each))

    for n in range(len(plans_template)):
        usage_temp[n] = np.random.normal(plans_mb[n], standard_deviations[n], n_each)

    usages = usage_temp[0]

    for n in range(len(plans_template) - 1):
        usages = np.concatenate((usages, usage_temp[n + 1]))

    plans = np.repeat(plans_template, n_each)
    '''
    # Display
    data = ((usages[:n_each], plans[:n_each]), 
        (usages[n_each + 1:n_each * 2], plans[n_each + 1:n_each * 2]), 
        (usages[n_each * 2 + 1: n_each * 3], plans[n_each * 2 + 1: n_each * 3]), 
        (usages[n_each * 3 + 1: n_each * 4], plans[n_each * 3 + 1: n_each * 4]), 
        (usages[n_each * 4 + 1: n_each * 5], plans[n_each * 4 + 1: n_each * 5]),
        (usages[n_each * 5 + 1: n_each * 6], plans[n_each * 5 + 1: n_each * 6]))
    colors = ("red", "blue", "yellow", "green", "purple", "black")
    groups = (str(plans_mb[0]), str(plans_mb[1]), str(plans_mb[2]), str(plans_mb[3]), str(plans_mb[4]), str(plans_mb[5]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show() #Uncomment to display

    # Remove to train
    exit(0)
    '''

    # Randomize the data
    X = np.vstack((usages, plans))
    K = X.T
    np.random.shuffle(K)

    usages = np.ravel(K.T[0])
    usages /= 2500 # Normalize the data. Took accuracy from 0.16 to 0.97
    plans = np.ravel(K.T[1])

    training_fraction = 0.7

    X_train = usages[ : (int)(training_fraction * n_each * len(plans_template))]
    Y_train = plans[ : (int)(training_fraction * n_each * len(plans_template))]
    X_test = usages[(int)(training_fraction * n_each * len(plans_template)) : ]
    Y_test = plans[(int)(training_fraction * n_each * len(plans_template)) : ]

    X_train = X_train.reshape((int)(training_fraction * n_each * len(plans_template)), 1)
    X_test = X_test.reshape((int)((1 - training_fraction) * n_each * len(plans_template)), 1)


    #logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/scalars/"
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    # Later go to logs folder, and type "Tensorboard --logdir scalars/", then type the url  http://GHOST:6006 in a browser.

    # To one-hot
    classes = 6
    Y_train = keras.utils.to_categorical(Y_train, classes)
    Y_test = keras.utils.to_categorical(Y_test, classes)

    #model = Sequential([ Dense(hidden_neurons, input_dim=input_size), Activation('sigmoid'), Dense(classes), Activation('softmax') ])

    model = Sequential();
    model.add(Dense(hidden_neurons, activation='sigmoid', input_dim=input_size))
    model.add(Dense(classes, activation='softmax'))

    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=True)
    # Or use optimizer='sdg' in model.compile to use default parameters

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

    model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          callbacks=[tensorboard_callback])

    score = model.evaluate(X_test, Y_test, verbose=1) 
    print('Test accuracy:', score[1])

    model.save(name + ".h5")

networks = {1: "mtn", 2: "airtel", 3: "glo", 4: "etisalat"}
durations = {1: "daily", 2: "weekly", 3: "monthly"}
net_du = {}

for i in networks:
    for j in durations:
        net_du[networks[i] + "_" + durations[j]] = networks[i] + "_" + durations[j]

#1
x = 1
y = 3
plans = {}
plans[networks[x] + "_" + durations[y]] = np.array([1500, 2000, 3500, 6500, 11000, 25000])
plans_template = {}
plans_template[networks[x] + "_" + durations[y]] = np.array([0, 1, 2, 3, 4, 5])
standard_deviations = {}
standard_deviations[networks[x] + "_" + durations[y]] = np.array([50, 60, 300, 350, 300, 2500])

#2
# TODO

n_each = 1000
epochs = 450

print("Choose network.\n (1)", networks[1].upper(), "(2) ", networks[2].upper(), "(3)", networks[3].upper(), "(4)", networks[4].upper(), "\n")
network = int(input())

print("Choose duration.\n (1)", durations[1], "(2) ", durations[2], "(3)", durations[3], "\n")
duration = int(input())
name = net_du[networks[network] + "_" + durations[duration]]

print("(0) Train (1) Test\n")
choice = int(input())

if choice == 0:
    train(name=name, plans_mb = plans[networks[network] + "_" + durations[duration]],
         plans_template = plans_template[networks[network] + "_" + durations[duration]],
        standard_deviations = standard_deviations[networks[network] + "_" + durations[duration]], 
        n_each = n_each, epochs = epochs)
else:
    if (not path.exists(name + ".h5")):
        print("There is no model to load!")
        exit(0)

    model = load_model('model.h5')

    plans_MB = plans[networks[network] + "_" + durations[duration]]
    usage_s = ""
    print("Enter the letter \'e\' to exit\n");

    while True:
        if  usage_s == "e":
            exit(0)

        usage_s = input("Enter Usage: ")
        usage = float(usage_s)
        usage /= 2500
        pred = model.predict([usage])
        print("You should use the", plans_MB[pred.argmax()], "MB data plan.\n\n")
