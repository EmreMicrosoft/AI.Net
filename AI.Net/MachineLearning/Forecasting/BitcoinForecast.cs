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
                trainSize: 496, // TODO: get data count here.
                seriesLength: 31,
                windowSize: 8,
                horizon: 4,
                confidenceLevel: 0.5f,
                confidenceLowerBoundColumn: nameof(BitcoinForecastModel.ConfidenceLowerBound),
                confidenceUpperBoundColumn: nameof(BitcoinForecastModel.ConfidenceUpperBound)
            )
            .Append(context.Transforms
                .Concatenate(outputColumnName: "ValueFutures",
                    new[]
                    {
                        nameof(BitcoinDataModel.Open),
                        nameof(BitcoinDataModel.High),
                        nameof(BitcoinDataModel.Low),
                        nameof(BitcoinDataModel.Close)
                    }))
            .Append(context.Transforms
                .Concatenate(outputColumnName: "VolumeFutures",
                    new[]
                    {
                        nameof(BitcoinDataModel.BaseVolume),
                        nameof(BitcoinDataModel.QuoteVolume),
                        nameof(BitcoinDataModel.TradeCount),
                        nameof(BitcoinDataModel.TakerBuyBaseVolume),
                        nameof(BitcoinDataModel.TakerBuyQuoteVolume)
                    }))
            .Append(context.Transforms
                .Concatenate(outputColumnName: "Futures",
                    new[] { "ValueFutures", "VolumeFutures" }
                    ));

        var trainedModel = pipeline.Fit(dataView);
        var forecastingEngine = trainedModel.CreateTimeSeriesEngine<BitcoinDataModel, BitcoinForecastModel>(context);
        return forecastingEngine.Predict();
    }
}