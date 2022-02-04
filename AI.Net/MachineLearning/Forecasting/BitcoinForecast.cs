using AI.Net.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI.Net.MachineLearning.Forecasting;

public class BitcoinForecast
{
    public static void Run()
    {
        var context = new MLContext();
        var dataView = context.Data
            .LoadFromTextFile<BitcoinDataModel>("DataSet/btcusdt.csv",
                hasHeader: true, separatorChar: ',');

        var pipeline = context.Forecasting
            .ForecastBySsa(
                outputColumnName: "PredictedValues",
                inputColumnName: nameof(BitcoinDataModel.Close),
                trainSize: 490,
                seriesLength: 14,
                windowSize: 10,
                horizon: 10,
                confidenceLevel: 0.98f,
                confidenceLowerBoundColumn: "LowerBoundValues",
                confidenceUpperBoundColumn: "UpperBoundValues"
            );

        var trainedModel = pipeline.Fit(dataView);
        var forecastingEngine = trainedModel.CreateTimeSeriesEngine<BitcoinDataModel, BitcoinForecastModel>(context);
        var forecasts = forecastingEngine.Predict();
    }
}