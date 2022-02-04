using AI.Net.Models;
using Microsoft.ML;

namespace AI.Net.MachineLearning.Forecasting;

public class BitcoinForecast
{
    public static void Run()
    {
        var context = new MLContext();
        var dataView = context.Data
            .LoadFromTextFile<BitcoinInputModel>("DataSet/btcusdt.csv",
                hasHeader: true, separatorChar: ',');

        var pipeline = context.Forecasting
            .ForecastBySsa(
                outputColumnName: "PredictedValues",
                inputColumnName: "Close",
                trainSize: 490,
                seriesLength: 14,
                windowSize: 10,
                horizon: 10,
                confidenceLevel: 0.98f,
                confidenceLowerBoundColumn: "LowerBoundValues",
                confidenceUpperBoundColumn: "UpperBoundValues"
            );
    }
}