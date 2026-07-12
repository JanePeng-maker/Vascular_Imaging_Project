import matplotlib.pyplot as plt
import os

# 自动获取plot.py自身目录
SCRIPT_PATH = os.path.abspath(__file__)
PLOT_ROOT = os.path.dirname(SCRIPT_PATH)
FIX_PLOT_DIR = os.path.join(PLOT_ROOT, "result", "plot")
os.makedirs(FIX_PLOT_DIR, exist_ok=True)

def loss_plot(args, loss):
    x = list(range(args.epoch))
    plt.figure()
    plt.plot(x, loss, label='Train Loss')
    plt.legend()
    save_name = f"{args.arch}_{args.batch_size}_{args.dataset}_{args.epoch}_loss.jpg"
    save_path = os.path.join(FIX_PLOT_DIR, save_name)
    plt.savefig(save_path)
    plt.close()

def metrics_plot(arg, name, *args):
    x = list(range(arg.epoch))
    names = name.split('&')
    plt.figure()
    for idx, data in enumerate(args):
        plt.plot(x, data, label=names[idx])
    plt.legend()
    save_name = f"{arg.arch}_{arg.batch_size}_{arg.dataset}_{arg.epoch}_{name}.jpg"
    save_path = os.path.join(FIX_PLOT_DIR, save_name)
    plt.savefig(save_path)
    plt.close()