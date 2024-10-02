import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from src.tasks.gsn2.structures import get_mnist_data, crop_insignificant_values

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = get_mnist_data()

TRAIN_DIGITS = [
    crop_insignificant_values(digit) / 255.0
    for digit_index, digit in enumerate(mnist_x_train[:10000])
]

for cropped_digit, digit in zip(TRAIN_DIGITS,  mnist_x_train[:10000]):
    # Normalize digit
    digit = digit / 255.0

    # Plot the images on adjacent axes
    fig, ax = plt.subplots(1, 2)

    # Display the digits
    ax[0].imshow(cropped_digit)
    ax[0].set_xlabel(str(cropped_digit.shape))
    ax[1].imshow(digit)
    ax[1].set_xlabel(str(digit.shape))


    plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
    plt.close()  # Close the current figure after keypress/mouse click
