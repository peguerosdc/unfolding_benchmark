import matplotlib
from matplotlib.ticker import ScalarFormatter

# https://github.com/karthik/wesanderson
colors = {
    "red": "#C93312",
    "grey": "#899DA4",
    "green": "#00A08A",
    "orange": "#F98400",
    "blue": "#5BBCD6",
}

font_sizes = {
    "middle": (18, 12),
}


def setup():
    # setup axis
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["axes.axisbelow"] = True
    # set up font
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["TeX Gyre Pagella"]
    # set default colors
    cycle = [colors[k] for k in colors]
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=cycle)
    # set default imshow
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    matplotlib.rcParams["image.cmap"] = "YlGnBu"
    # remove white margins
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.transparent"] = True


def setup_font_size(size=font_sizes["middle"]):
    main, sub = size
    matplotlib.rc("axes", titlesize=main)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=main)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=sub)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=sub)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=main)  # legend fontsize


def setup_font_1x2():
    matplotlib.rc("axes", titlesize=40)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=40)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=30)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=30)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=26)  # legend fontsize


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.2f"  # Give format here


def get_sci_decimal_format():
    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))
    return yfmt


def get_1x1_im_figure():
    matplotlib.rc("axes", titlesize=18)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=18)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=12)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=12)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=18)  # legend fontsize
    fig, ax = matplotlib.pylab.subplots(1, 1, figsize=(10, 7))
    return fig, ax