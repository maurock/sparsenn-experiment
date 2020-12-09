import matplotlib.pyplot as plt

# Plot 3D function
def plot_3D(x,y,z,title=None,show=True):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    if title!=None:
        plt.title(title)
    if show:
        plt.show()

def plot_2D(x,z,c=None,title=None,legend=[],show=True):   
    # Plot lateral view
    plt.plot(x, z, c=c)
    if title!=None:
        plt.title(title)
    if show:
        plt.legend(legend)
        plt.show()        

def scatter_2D(x,z,c=None,title=None,legend=[],show=True):   
    # Plot lateral view
    plt.scatter(x, z, s=10, c=c)
    if title!=None:
        plt.title(title)
    if show:
        plt.legend(legend)
        plt.show() 