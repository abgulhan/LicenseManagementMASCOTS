#%%
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
from pathlib import Path
import os

from math import sqrt
import multiprocessing as mp
from copy import deepcopy
import openpyxl

import mxnet as mx
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.mx import DeepAREstimator, Trainer, MQCNNEstimator, MQRNNEstimator, DeepStateEstimator, DeepVAREstimator
from gluonts.ext import prophet
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from typing import Optional, Tuple, Iterator
from gluonts.dataset.common import DataEntry, Dataset



# user file imports
import utils

import gluonts

# argparse correction for boolean values. From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## Forecasting
    parser.add_argument('--data_file', type=str, default="./processed/comsol2015/combined_1day_crop.csv", help='file containing data to train & test model')
    parser.add_argument('--start', type=str, default=None, help='Start Date of data to plot, inclusive. Format: %Y-%m-%d Default starts at beginning')
    parser.add_argument('--end', type=str, default=None, help='Finish Date of data to plot, inclusive. Default finishes at end')

    parser.add_argument('--model', type=str, default='mqrnn', help='Name of model to use. Args: deepar, mqcnn, mqrnn')

    parser.add_argument('--target', type=str, default="COMSOL", help='Feature name to forecast')
    parser.add_argument('--freq', type=str, default="D", help='Frequency of data set. Ex: 0D, H, min. For more information see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases')
    parser.add_argument('--remove_feat_thresh', type=float, default=0.9, help='Remove features used under this threshold')
    parser.add_argument('--feature_select', type=str2bool, default=True, help="Perform feature selection on data")
    parser.add_argument('--scaling', type=str2bool, default=True, help="Perform 0-1 on dynamic features")
    parser.add_argument('--use_dynamic_real', type=str2bool, default=True, help='use dynamic (mutivariate) features when training model')
    parser.add_argument('--pred_len', type=int, default=360, help='How far to forecast')
    parser.add_argument('--pred_count', type=int, default=3, help='How many forecasts to make')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs in training')
    parser.add_argument('--context_length', type=int, default=None, help='Number of steps to input before making prediction. Default pred_len')
    parser.add_argument('--average_window', type=int, default=None, help='Size of moving average window to apply to target. Default not used.')
    parser.add_argument('--average_features', type=str2bool, default=False, help='Whether to apply moving average to all features. Default only apply to target feature.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size when training. Decrease if running out of GPU memory')
    parser.add_argument('--add_time_feat', type=str2bool, default=True, help="Whether to add derived time features.")

    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers in model (if supported as a parameter)')
    parser.add_argument('--num_cells', type=int, default=None, help='Number of cells in model (if supported as a parameter)')
    


    parser.add_argument('--save_model', type=str, default=None, help='DIRECTORY location to save model. Makes multiple files in directory. Does not save if omitted.')
    parser.add_argument('--load_model', type=str, default=None, help='DIRECTORY location to load saved model.')
    parser.add_argument('--plot', type=str2bool, default=True, help="Plot results")

    # MQ-CNN arguments
    parser.add_argument('--quantiles', type=str, default='0.05,0.5,0.95', help='Quantiles to predict. Only used in MQCNN model. Separate quantiles with ,')

    # DeepVAR arguments
    #parser.add_argument('--plot_target', type=str, default='COMSOL', help='Which feature/license to plot in a many-to-many forecast.')


    return parser.parse_args()


from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import period_index
from gluonts.dataset.common import DataEntry
from typing import Tuple

def _to_dataframe(input_label: Tuple[DataEntry, DataEntry]) -> pd.DataFrame:
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)

def make_evaluation_predictions_(test_data: Dataset, predictor: Predictor, num_windows: int, filter_size: int = 0) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    window_length = predictor.prediction_length + predictor.lead_time
    test_data = test_gen.generate_instances(prediction_length=pred_len, windows=num_windows)    
    if filter_size == 0:
        return (
            predictor.predict(test_data.input, num_samples=1),
            map(_to_dataframe, test_data),
        )
    else:
        input_rolling_avg = list(test_data.input)
        print(input_rolling_avg)
        print(type(input_rolling_avg[0]))
        
        return (
            predictor.predict(input_rolling_avg, num_samples=1), ### TODO apply filter on test_data.input
            map(_to_dataframe, test_data),
        )
def plot_prob_forecasts(ts_entry, forecast_entry, plot_length, prediction_intervals, pred_name=None):
    #plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    if pred_name == 'mqcnn':
        forecast_entry.plot(color="g")
    else:
        pass
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    
    wb = openpyxl.load_workbook('input.xlsx')
    
    plt.show()    


def dataentry_to_dataframe(entry):
    df = pd.DataFrame(
        entry["target"],
        columns=[entry.get("item_id")],
        index=pd.period_range(
            start=entry["start"], periods=len(entry["target"]), freq=entry["start"].freq
        ),
    )

    return df


class DeepARTuningObjective:
    def __init__(
        self, dataset, prediction_length, freq, metric_type="mean_wQuantileLoss"
    ):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.freq = freq
        self.metric_type = metric_type

        self.train, test_template = split(dataset, offset=-self.prediction_length)
        validation = test_template.generate_instances(
            prediction_length=prediction_length
        )
        self.validation_input = [entry[0] for entry in validation]
        self.validation_label = [
            dataentry_to_dataframe(entry[1]) for entry in validation
        ]

    def get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 10, 50),
        }

    def __call__(self, trial):
        params = self.get_params(trial)
        estimator = DeepAREstimator(
            num_layers=params["num_layers"],
            hidden_size=params["hidden_size"],
            prediction_length=self.prediction_length,
            freq=self.freq,
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 10,
            },
        )

        predictor = estimator.train(self.train, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)

        forecasts = list(forecast_it)

        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts, num_series=len(self.dataset)
        )
        return agg_metrics[self.metric_type]

if __name__ == '__main__':
    args = arg_parse()    

    data_file = args.data_file
    data = pd.read_csv(data_file, index_col=0, parse_dates=[0])
    data = utils.crop_data(data, args.start, args.end)

    remove_thresh = args.remove_feat_thresh
    remove_unused_features = (remove_thresh!=1)
    remaining_features=list(data.columns)
    print(f"all features len {len(remaining_features)}:\n {remaining_features}")
    
    # remove rarely used features
    if remove_unused_features:
        data = data.drop(columns=data.columns[data.eq(0).mean()>remove_thresh]) 
        remaining_features = list(data.columns)
        print(f"remaining_feature above threshold, len {len(remaining_features)}:\n {remaining_features}")    
    
    
    #plotting.get_line_graph(data, plt_type='plotly')
    

    
    print("Number GPU's: " + str(mx.context.num_gpus()))
    target_feature_name = args.target
    dynamic_feature_names = []
    use_dynamic_real = args.use_dynamic_real
    
    #if not use_dynamic_real:
    #    data = data[args.target]
    if args.feature_select:
        ignore_features = ["SERIAL", "CLUSTERNODE"] # TODO USE FEATURE SELECTION
    else:
        ignore_features = []
        
    for feature in remaining_features:
        if feature != target_feature_name and feature not in ignore_features:
            dynamic_feature_names.append(feature)
            
    # scaling
    do_scaling = args.scaling
    if do_scaling:
        a, b = 0, 1
        for feature in dynamic_feature_names:
            #print(data[feature])
            x, y = data[feature].min(), data[feature].max()
            data[feature] = (data[feature] - x) / (y - x) * (b - a) + a
    
    print("dynamic features:")
    print(dynamic_feature_names)
    #print(f"inferred freq {pd.infer_freq(data)}")

    # apply moving average # TODO deal with NaN values at beginning and end
    data_original = None
    if args.average_window != None and args.average_window > 0:
        data_original = data[target_feature_name].copy()
        if args.average_features: # apply to all features
            data = data.rolling(args.average_window, center=True).mean()
        else: # only apply to target feature
            data[target_feature_name] = data[target_feature_name].rolling(args.average_window, center=True).mean()

    # sanity check
    scheck = not use_dynamic_real
    if scheck:    
        for f in data.columns:
            if f != args.target:
                data.loc[:,f] = 0
                data[f] = 0
                
        print(data)
    
    if args.model.lower() == 'deepar' or args.model.lower() == 'deepar2' or args.model.lower() == 'deepar3':
        dataset = PandasDataset(data, target=target_feature_name, feat_dynamic_real=dynamic_feature_names,freq=args.freq, assume_sorted=True)
    elif args.model.lower() == 'deepvar':
        dataset = PandasDataset(data, target=[target_feature_name]+dynamic_feature_names, freq=args.freq, assume_sorted=True)        
    else:
        dataset = PandasDataset(data, target=target_feature_name, past_feat_dynamic_real=dynamic_feature_names,freq=args.freq, assume_sorted=True)

    print("Dataset:")
    print(dataset)
    # Train a DeepAR model on all data but the last 12 months
    pred_len = args.pred_len
    pred_count = args.pred_count # how many pred_len predictions to make
    #data.iloc[-pred_len*pred_count:] = 0 #############
    epochs = args.epochs
    training_data, test_gen = split(dataset, offset=-pred_len*pred_count)
    
    training_data = list(training_data)
    #print(training_data)
    #print(len(training_data[0]['target']))
    #print(f"length of train dataset {len(training_data)}")
    #print(f"length of test dataset {len(test_gen)}")
    if args.model.lower() == 'deepar':
        from gluonts.mx.distribution import NegativeBinomialOutput
        model = DeepAREstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length#,
                    #num_layers=2,
                    #num_cells=50,
                    #distr_output=NegativeBinomialOutput()
                )
    if args.model.lower() == 'deepar2':
        from gluonts.mx.distribution import NegativeBinomialOutput
        model = DeepAREstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length,
                    num_layers=4,
                    num_cells=100#,
                    #distr_output=NegativeBinomialOutput()
                )
    if args.model.lower() == 'deepar3':
        from gluonts.mx.distribution import NegativeBinomialOutput
        model = DeepAREstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length,
                    num_layers=6,
                    num_cells=150,
                    batch_size=args.batch_size
                    #distr_output=NegativeBinomialOutput()
                )
    elif args.model.lower() == 'deepar_pastfeat':
        # using PR from https://github.com/awslabs/gluonts/pull/1757 for past_feat_dynamic_real
        from gluonts.mx.distribution import NegativeBinomialOutput
        model = DeepAREstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_past_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length#,
                    #num_layers=2,
                    #num_cells=50,
                    #distr_output=NegativeBinomialOutput()
                )
    elif args.model.lower() == 'mqcnn':
        from gluonts.mx.distribution import NegativeBinomialOutput
        from gluonts.mx.distribution import StudentTOutput
        quantiles = args.quantiles.split(',')
        model = MQCNNEstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_past_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length,
                    quantiles=quantiles,
                    add_time_feature=args.add_time_feat,
                    add_age_feature=False,
                    enable_decoder_dynamic_feature=True, #???
                    batch_size=args.batch_size
                )
    elif args.model.lower() == 'mqcnn2':
        from gluonts.mx.distribution import NegativeBinomialOutput
        from gluonts.mx.distribution import StudentTOutput
        quantiles = args.quantiles.split(',')
        s = 45
        model = MQCNNEstimator(
                    prediction_length=pred_len, 
                    freq=args.freq, 
                    trainer=Trainer(epochs=epochs), 
                    use_past_feat_dynamic_real=use_dynamic_real,
                    context_length=args.context_length,
                    quantiles=quantiles,
                    add_time_feature=args.add_time_feat,
                    add_age_feature=False,
                    enable_decoder_dynamic_feature=True, #???
                    batch_size=args.batch_size,
                    embedding_dimension=[50],
                    decoder_mlp_dim_seq=[s,s],
                    channels_seq=[s,s],
                    dilation_seq=[1,5],
                    kernel_size_seq=[7,3]
                )
    elif args.model.lower() == 'mqrnn':
        model = MQRNNEstimator(
            prediction_length=pred_len, 
            freq=args.freq, 
            trainer=Trainer(epochs=epochs), 
            use_past_feat_dynamic_real=use_dynamic_real,
            context_length=args.context_length
        )
    elif args.model.lower() == 'prophet':
        pass
    elif args.model.lower() == 'mqf2':
        pass
    elif args.model.lower() == 'deepstate':
        model = DeepStateEstimator(
            prediction_length=pred_len,
            freq=args.freq,
            trainer=Trainer(epochs=epochs), 
            use_feat_dynamic_real=False,#use_dynamic_real,
            past_length=args.context_length,
            num_periods_to_train=4,
            batch_size=args.batch_size,
            use_feat_static_cat=False,
            cardinality=[0]
            
        )
    elif args.model.lower() == 'deepvar':
        print(f"Dimension: {len(list(data.columns))}")
        model = DeepVAREstimator(
            prediction_length=pred_len,
            context_length=args.context_length,
            freq=args.freq,
            target_dim=len(list(data.columns)),
            trainer=Trainer(epochs=epochs), 
            use_feat_dynamic_real=use_dynamic_real,
            batch_size=args.batch_size,
            num_layers = args.num_layers,
            num_cells=args.num_cells,
        )
    else:
        print("using default model DeepAR")
        model = DeepAREstimator(
            prediction_length=pred_len, 
            freq=args.freq, 
            trainer=Trainer(epochs=epochs), 
            use_feat_dynamic_real=use_dynamic_real,
            context_length=args.context_length
        )
    if args.load_model:
        from gluonts.model.predictor import Predictor
        print("Loading saved model")
        predictor = Predictor.deserialize(Path(args.load_model))  
    else:
        predictor = model.train(training_data)
        
    if args.save_model != None:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        predictor.serialize(Path(args.save_model))
    #%%
    # Generate test instances and predictions for them
    test_data = test_gen.generate_instances(prediction_length=pred_len, windows=pred_count)
    '''
    forecasts = list(predictor.predict(test_data.input))
    # Plot predictions
    data[target_feature_name].plot(color="black")
    for forecast, color in zip(forecasts, ["green", "blue", "purple"]*(-(-pred_count//3))):
        forecast.plot(color=f"tab:{color}")
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    plt.show()
    '''
    
    # Evaluate
    #from gluonts.evaluation import make_evaluation_predictions
    forecast_it, ts_it = make_evaluation_predictions_(
        test_data=dataset,  # test dataset
        predictor=predictor,  # predictor
        num_windows=args.pred_count  # number of windows to predict
    )
    forecasts = list(forecast_it)
    tss = list(ts_it) # ground truth for each forecast


    # Plot predictions    
    #print(len(forecasts))
    print('=====forecasts=====')
    print(forecasts)
    print(type(forecasts[0]))
    
    targets = [target_feature_name] + dynamic_feature_names # for use in many to many prediction (eg. deepVAR)
    if args.plot:
        if args.average_window != None and args.average_window > 0:
            data_original.plot(color="black")
        else:
            data[target_feature_name].plot(color="black")
        for forecast, color in zip(forecasts, ["green", "blue", "purple"]*(-(-pred_count//3))):
            if args.model.lower() == 'deepvar':
                for i, target_name in enumerate(targets):
                    if target_name != args.target:
                        continue
                    f = forecast.copy_dim(i)
                    f.plot(color=f"tab:{color}", label=target_name)
            else:
                forecast.plot(color=f"tab:{color}")
        plt.legend(["True values"], loc="upper left", fontsize="xx-large")
        quantiles = args.quantiles.split(',')

        ax = plt.gca() # get axis handle
        if args.model == 'mqcnn' and args.pred_count==1:
            dir_path = f"plot_data/{args.target}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plot_df = pd.DataFrame() 
            print(f"num lines {len(ax.lines)}")
            for i, line in enumerate(ax.lines):
                if i == 0: continue
                if i == 1:
                    plot_df[f"time"] = line.get_xdata()
                plot_df[f"quant_{quantiles[i-1]}_y"] = line.get_ydata()
            plot_df.to_csv(f"{dir_path}/{args.target}-quantiles-enddate-{args.end}.csv", index=False)
        plt.show()#block=False)
        #plt.pause(1)

    #print(f'forecasts_eval {forecasts}')
    #print(f'tss {tss}')

    import json
    from gluonts.evaluation import Evaluator
    evaluator = Evaluator(quantiles=quantiles)#[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    
    print(json.dumps(agg_metrics, indent=4))
    print(item_metrics.head())
    item_metrics.to_csv('./eval/tmp.csv')
    
    
    item_metrics.plot(x="MAPE", y="MSE", kind="scatter")
    plt.grid(which="both")
    plt.show()

