from generate_mask import *

def plot_2d_pdf(pdf):
    side_length = pdf.shape[0]
    xlist = np.linspace(0, 1, side_length)
    ylist = np.linspace(0, 1, side_length)
    X,Y = np.meshgrid(xlist, ylist)

    fig, ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y,pdf, levels=1000)
    fig.colorbar(cp)
    plt.show()


def plot_2d_mask(mask):
    side_length = mask.shape[0]
    xlist = np.linspace(0, 1, side_length)
    ylist = np.linspace(0, 1, side_length)
    X,Y = np.meshgrid(xlist, ylist)

    fig, ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y,mask, levels=2)
    fig.colorbar(cp)
    plt.show()

# an example
if __name__ == "__main__":
    kq_mask = generate_kq_power_density_mask(0.9, 0.2, 3, 12, 6, 4)
    mask = generate_power_density_mask(0.9, 0.2, 2, 70, 4)
    plot_2d_mask(mask)
