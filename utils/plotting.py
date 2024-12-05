import matplotlib.pyplot as plt

def multi_boxplot(data, keys, title, x_ticks, y_label, violin=False,
                  rotation=20, left=0.1, bottom=0.1,
                  **kwargs):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=left, bottom=bottom)
    for i, key in enumerate(keys):
        if violin:
            if len(data[key]) == 0:  # Edge case for empty data
                data[key][0] = 0
            ax.violinplot(data[key], positions=[i], showmedians=True)
        else:
            ax.boxplot(data[key], positions=[i], tick_labels=[key])
    if 'y_lim' in kwargs:
        ax.set_ylim(kwargs.get('y_lim'))
    if kwargs.get('log_y', False):
        ax.set_yscale('log')
    plt.axhline(0)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if 'x_label' in kwargs:
        ax.set_xlabel(kwargs.get('x_label'))
    if violin:
        ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=rotation)
    plt.show()
