if __name__ == '__main__':
    grid_search = args.grid_search
    if grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            parameters=yaml.load(open(os.path.join(config['config'], 'tcn_gs.yaml'))),
            db_name=config['db'],
            ex_name='tcn_grid_search',
            f_main=main,
            f_metrics=log_metrics,
            observer_type='mongodb')
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='tcn',
            f_main=main,
            f_config=os.path.join(config['config'], 'tcn.yaml'),
            f_metrics=log_metrics,
            observer_type='mongodb')