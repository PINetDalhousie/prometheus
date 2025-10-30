import matplotlib.pyplot as plt

def plot_distribution(data,path):
    plt.hist(data, bins='auto', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.savefig(path+'distribution_plot.png')  # Save the plot as an image
    plt.close()

if __name__ == '__main__':
    pass