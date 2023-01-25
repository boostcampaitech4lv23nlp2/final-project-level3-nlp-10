# import sys
from model.dpr_eval import eval as dpr_eval

# from model.dpr_train import train as dpr_train
from omegaconf import OmegaConf

if __name__ == "__main__":
    """
    cmd = sys.argv[1]
    if len(sys.argv) > 2:
        args = sys.argv[2:]
    if cmd == 'dpr_train':
        dpr_train()
    """
    conf = OmegaConf.load("./config.yaml")
    # dpr_train(conf)
    dpr_eval(conf)
