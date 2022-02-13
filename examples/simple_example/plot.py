from matplotlib import pyplot as plt


def drawPlot(iteration, log):
    location = log.getColumn("location")
    battery = log.getColumn("battery")
    futureBattery = log.getColumn("future_battery")

    fig, ax1 = plt.subplots()
    plt.title(f"Iteration: {iteration}")
    ax1.set_xlabel('Step')

    ax1.set_ylabel('Battery')
    l1 = ax1.plot(battery, color='tab:green', label='Battery')
    l2 = ax1.plot(futureBattery, color='tab:blue', label='Future battery')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Location')
    l3 = ax2.plot(location, color='tab:red', label='Location')

    ls = l1 + l2 + l3
    plt.legend(ls, [l.get_label() for l in ls])

    fig.tight_layout()
    plt.show()
