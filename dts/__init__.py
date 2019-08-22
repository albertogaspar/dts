import yaml, os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_file = os.path.join(base_dir, 'config.yaml')
config = yaml.load(open(config_file))
for k,v in config.items():
    if k != 'db':
        config[k] = os.path.join(base_dir, v)

from dts.utils.logger import logger