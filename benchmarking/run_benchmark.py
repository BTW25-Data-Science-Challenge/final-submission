
from benchmarking.benchmarker import BenchmarkMaker


if __name__ == '__main__':
    bench_config = {'ground_truth_file_name': 'day_ahead_prices.csv',
                    'input_dir': 'input',
                    'export_dir': 'results'}
    BenchMaker = BenchmarkMaker(config=bench_config)

    BenchMaker.plot_compare_predictions()
    pass
