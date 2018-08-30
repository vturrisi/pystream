import pandas as pd
import sys
import os
sys.path.append('..')

from src import (VFDT, SVFDT, SVFDT_II,
                 OzaBagging, OzaBoosting,
                 LeveragingBagging,
                 OnlineAccuracyUpdatedEnsemble,
                 AdaptiveRandomForests,
                 read_arff_meta,
                 instance_gen, EvaluatePrequential)

def run():
    DEBUG = True
    # DEBUG = False
    commands = [('datasets/elecNormNew.csv', 'log/elecNorm')]

    for fname, output_folder in commands:
        dataset_name = fname.split('/')[-1].split('.csv')[0]
        meta_file = f'datasets/metas/{dataset_name}.meta'
        dtype, types, classes = read_arff_meta(meta_file)
        n_classes = len(classes)
        only_binary_splits = False
        base_learners_n_args = [
                                ('vfdt', VFDT,
                                 {'gp': 400, 'split_criterion': 'infogain',
                                  'tiebreaker': 0.05,
                                  'only_binary_splits': only_binary_splits}),

                                # ('svfdt', SVFDT,
                                #  {'gp': 400, 'split_criterion': 'infogain',
                                #   'tiebreaker': 0.05,
                                #   'only_binary_splits': only_binary_splits}),
                                #
                                # ('svfdt_ii', SVFDT_II,
                                #  {'gp': 400, 'split_criterion': 'infogain',
                                #   'tiebreaker': 0.05,
                                #   'only_binary_splits': only_binary_splits}),
                                ]

        for name, base_learner, kwargs in base_learners_n_args:
            for iteration in range(1):
                algorithm = base_learner(types, n_classes, **kwargs)
                # log_file = f'test_time.csv'
                log_file = None
                evaluator = EvaluatePrequential(n_classes, algorithm,
                                                algorithm_type='tree')
                stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                evaluator.train_test_prequential_cm(stream, DEBUG,
                                                    1000,
                                                    log_file=log_file)
                print(name, evaluator.stats.accuracy)
                print(name, 'train', evaluator.stats['train_time'])
                print(name, algorithm._stats['n_nodes'])
                print(name, algorithm.memory_size())
                print()

                # OzaBag
                # algorithm = OzaBagging(types, n_classes,
                #                              n_predictors=10,
                #                              base_learner=base_learner,
                #                              base_learner_kwargs=kwargs)
                # log_file = None
                # evaluator = EvaluatePrequential(n_classes, algorithm,
                #                                 algorithm_type='tree')
                # stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                # # evaluator.train_test_prequential_cm(stream, DEBUG,
                # #                                     1000,
                # #                                     log_file=log_file)
                # print(name, evaluator.stats.accuracy)
                # print(name, 'train', evaluator.stats['train_time'])
                # print(name, algorithm._stats['n_nodes'])
                # print(name, algorithm.memory_size())
                # print()

                # # OzaBoost
                # algorithm = OzaBoosting(types, n_classes,
                #                               n_predictors=10,
                #                               base_learner=base_learner,
                #                               base_learner_kwargs=kwargs)
                # log_file = None
                # evaluator = EvaluatePrequential(n_classes, algorithm,
                #                                 algorithm_type='tree')
                # stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                # # evaluator.train_test_prequential_cm(stream, DEBUG,
                # #                                     1000,
                # #                                     log_file=log_file)
                # print(name, evaluator.stats.accuracy)
                # print(name, 'train', evaluator.stats['train_time'])
                # print(name, algorithm._stats['n_nodes'])
                # print(name, algorithm.memory_size())
                # print()

                # # LevBag
                # algorithm = LeveragingBagging(types, n_classes,
                #                                     n_predictors=10,
                #                                     base_learner=base_learner,
                #                                     base_learner_kwargs=kwargs)
                # log_file = None
                # evaluator = EvaluatePrequential(n_classes, algorithm,
                #                                 algorithm_type='tree')
                # stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                # # evaluator.train_test_prequential_cm(stream, DEBUG,
                # #                                     1000,
                # #                                     log_file=log_file)
                # print(name, evaluator.stats.accuracy)
                # print(name, 'train', evaluator.stats['train_time'])
                # print(name, algorithm._stats['n_nodes'])
                # print(name, algorithm.memory_size())
                # print()
                #
                # # OAUE
                #
                # algorithm = OnlineAccuracyUpdatedEnsemble(types, n_classes,
                #                                                 n_predictors=10,
                #                                                 window_size=1000,
                #                                                 base_learner=base_learner,
                #                                                 base_learner_kwargs=kwargs)
                # log_file = None
                # evaluator = EvaluatePrequential(n_classes, algorithm,
                #                                 algorithm_type='tree')
                # stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                # # evaluator.train_test_prequential_cm(stream, DEBUG,
                # #                                     1000,
                # #                                     log_file=log_file)
                # print(name, evaluator.stats.accuracy)
                # print(name, 'train', evaluator.stats['train_time'])
                # print(name, algorithm._stats['n_nodes'])
                # print(name, algorithm.memory_size())
                # print()
                #
                # # ARF
                #
                # algorithm = AdaptiveRandomForests(types, n_classes,
                #                                         warning_delta=0.01,
                #                                         drift_delta=0.001,
                #                                         n_predictors=10,
                #                                         base_learner=base_learner,
                #                                         base_learner_kwargs=kwargs)
                # log_file = None
                # evaluator = EvaluatePrequential(n_classes, algorithm,
                #                                 algorithm_type='tree')
                # stream = instance_gen(fname, dtype)
                # evaluator.train_test_prequential_no_partial_cm(stream, DEBUG,
                #                                                1000,
                #                                                log_file=log_file)
                # # evaluator.train_test_prequential_cm(stream, DEBUG,
                # #                                     1000,
                # #                                     log_file=log_file)
                # print(name, evaluator.stats.accuracy)
                # print(name, 'train', evaluator.stats['train_time'])
                # print(name, algorithm._stats['n_nodes'])
                # print(name, algorithm.memory_size())
                # print()


if __name__ == '__main__':
    run()
