# import sys

from model.dpr_train import train as dpr_train

if __name__ == "__main__":
    """
    cmd = sys.argv[1]
    if len(sys.argv) > 2:
        args = sys.argv[2:]
    if cmd == 'dpr_train':
        dpr_train()
    """
    dpr_train()
