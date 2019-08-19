from dts.utils.utils import get_args
from dts.utils.losses import r2, smape, nrmse_a, nrmse_b, nrmse_c, nrmsd
from dts.utils.experiments import DTSExperiment, log_metrics, run_single_experiment, run_grid_search

metrics = ['mse',
           'mae',
           nrmse_a,
           nrmse_b,
           nrmsd,
           r2,
           smape,
           'mape']