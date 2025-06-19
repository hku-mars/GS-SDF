# read output/mem_usage.txt and draw the memory usage curve
# Usage: python3 draw_mem.py

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path


def draw(args):
    log_file = args.log_file
    log_file = Path(log_file[0])
    log_file_name = log_file.name.replace(".txt", "")
    # make directory to save the loss curve
    save_fig_paths = log_file.parent / log_file_name
    save_fig_paths.mkdir(exist_ok=True)
    
    linestyle = "-"
    linewidth = 1
    marker = "o"
    markersize = 1

    # read output/mem_usage.txt and
    titles = []
    x = []
    with open(log_file, "r") as f:
        for title in f.readline().split():
            titles.append(title)
            x.append([])

        for line in f:
            line = line.strip()
            if line and (len(line.split()) == len(titles)):
                j = 0
                for attr in line.split():
                    x[j].append(float(attr))
                    j = j + 1

    iter_range = list(range(len(x[0])))
    for j in range(len(titles)):
        # find the min max to normalize the loss
        # min_loss = min(x[j + 1])
        # max_loss = max(x[j + 1])
        # loss = [((i - min_loss) / (max_loss - min_loss)) for i in x[j + 1]]

        loss = x[j]
        plt.cla()
        plt.plot(iter_range, loss, label=titles[j])

        plt.legend(prop={"size": 11})
        if args.vis:
            plt.show()
        # plt.show()
        plt.savefig(log_file.parent / log_file_name / (titles[j] + ".png"))
    # plt.plot(x, cpu_mem, "-.", label="System Memory Usage", color="red")

    # plt.plot(x, gpu_mem, "-.", label="Video Memory Usage", color="blue")
    # plt.xlabel("Frame Index", fontdict={"size": 13})
    # plt.ylabel("Memory Usage (GB)", fontdict={"size": 13})
    # plt.ylim(ymin=0, ymax=12)
    # plt.xlim(xmin=0, xmax=4540)

    # plt.legend(prop={"size": 11})
    # plt.show()
    # plt.savefig(model_paths / "loss.png", dpi=300)


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--log_file", "-l", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--vis", "-v", type=bool, default=False)
    args = parser.parse_args()
    draw(args)
