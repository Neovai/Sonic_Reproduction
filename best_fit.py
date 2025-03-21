import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy.optimize import curve_fit

def convert_to_secs(time_str: str):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], time_str.split(":")))

def get_info(file_name):
    """
    returns x, y, time, name of method
    """
    dataframe = pd.read_table(file_name, sep=" ")
    print(dataframe)
    data = dataframe.to_numpy()
    ami_scores = data[:, 4].astype(np.float32)
    s_vals = data[:, 1].astype(np.float32)
    method = data[0, 0]
    eps_l2 = data[:, 5].astype(np.int32)
    times = data[:, -1]
    # convert time to seconds:
    secs = np.vectorize(convert_to_secs)
    times = secs(times)

    return eps_l2, ami_scores, s_vals, times, method

def get_pred(x, y):
    c, b, a = np.polyfit(x, y, deg=2)
    y_predicted = c * x**2 + b * x + a
    return y_predicted

if __name__ == "__main__":
    dataset = "FASHIONMNIST_DressBoot"
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    x1, y1, s1, times1, method1 = get_info(file1)
    x2, y2, s2, times2, method2 = get_info(file2)

    # get line of best fit:
    y1_fit = get_pred(x1, y1)
    y2_fit = get_pred(x2, y2)

    # # Generate the line of best fit
    # polynomial = np.polynomial.polynomial(coefficients)

    # # Create the plot for AMI scores:
    plt.figure()
    plt.scatter(x1, y1, label=f'{method1}', color="g")
    plt.scatter(x2, y2, label=f'{method2}', color="b")
    # plt.scatter(x1, y1_fit, linewidths=2, color="orange")
    # plt.scatter(x2, y2_fit, linewidths=2, color="r")
    plt.xlabel('epsilon L2')
    plt.ylabel('AMI')
    plt.title(f"{method1} vs. {method2} on {dataset}")
    plt.legend()
    plt.savefig(f"{dataset}_effect.png")

    # Create the plot for times:
    plt.figure()
    plt.plot(s1, times1, label=f'{method1}', color="gray")
    plt.plot(s2, times2, label=f'{method2}', color="b")
    # plt.scatter(x1, y1_fit, linewidths=2, color="orange")
    # plt.scatter(x2, y2_fit, linewidths=2, color="r")
    plt.xlabel('s: Portion of poisoned dataset')
    plt.ylabel('Time (s)')
    plt.title(f"{method1} vs. {method2} on {dataset}")
    plt.legend()
    plt.savefig(f"{dataset}_time.png")