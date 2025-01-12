
from benchmarking.benchmarker import BenchmarkMaker


if __name__ == '__main__':
    bench_config = {'ground_truth_file_name': 'day_ahead_prices.csv',
                    'input_dir': 'input',
                    'export_dir': 'results',
                    'tz': 'Europe/Brussels'}
    BenchMaker = BenchmarkMaker(config=bench_config)

    BenchMaker.plot_compare_predictions()
    BenchMaker.plot_rmse_per_hour()
    BenchMaker.plot_rmse_per_day()
    BenchMaker.plot_mae_per_hour()
    BenchMaker.plot_mae_per_day()
    pass
