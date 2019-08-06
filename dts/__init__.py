import yaml, os

config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.yaml')
config = yaml.load(open(config_file))

from dts.utils.logger import logger