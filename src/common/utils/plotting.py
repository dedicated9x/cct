import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' for non-interactive plots
import matplotlib.pyplot as plt

import numpy as np
import cv2
import types

def plt_show_fixed(figure: types.ModuleType):
    # Save the plot as a temporary image file
    plot_path = 'temp_plot.png'
    plt.savefig(plot_path)

    # Step 2: Load the image with OpenCV
    image = cv2.imread(plot_path)

    # Optionally, delete the temporary image file
    import os
    os.remove(plot_path)

    # Display the image using OpenCV
    cv2.imshow('Plot', image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example(case: str):
    if case == "plt":
        # Step 1: Generate the plot using matplotlib
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label='Sine Wave')
        plt.title('Simple Sine Wave')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
    elif case == "fig,ax":
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(x, y, label='Sine Wave')

        ax.set_title('Simple Sine Wave')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
    else:
        raise NotImplementedError

    plt_show_fixed(plt)


if __name__ == '__main__':
    # example(case="plt")
    example(case="fig,ax")