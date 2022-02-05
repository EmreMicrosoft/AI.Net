using AI.Net.Models;
using AI.Net.Utilities;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace AI.Net.MachineLearning.Forecasting;

public class BitcoinForecast
{
    public static BitcoinForecastModel GetResults()
    {
        var context = new MLContext();
        var dataView = context.Data
            .LoadFromTextFile<BitcoinDataModel>($"{Constants.SolutionPath()}\\DataSet\\btcusdt.csv",
                hasHeader: true, separatorChar: ',');

        var pipeline = context.Forecasting
            .ForecastBySsa(
                outputColumnName: nameof(BitcoinForecastModel.PredictedValues),
                inputColumnName: nameof(BitcoinDataModel.Close),
                trainSize: 490, // TODO: get data count here.
                seriesLength: 35,
                windowSize: 10,
                horizon: 5,
                confidenceLevel: 0.99f,
                confidenceLowerBoundColumn: nameof(BitcoinForecastModel.ConfidenceLowerBound),
                confidenceUpperBoundColumn: nameof(BitcoinForecastModel.ConfidenceUpperBound)
            );
            //.Append();

        var trainedModel = pipeline.Fit(dataView);
        var forecastingEngine = trainedModel.CreateTimeSeriesEngine<BitcoinDataModel, BitcoinForecastModel>(context);
        return forecastingEngine.Predict();
    }
}