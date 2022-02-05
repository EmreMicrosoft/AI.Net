namespace AI.Net.Models;

public class BitcoinForecastModel
{
    public float[] PredictedValues { get; set; }
    public float[] ConfidenceLowerBound { get; set; }
    public float[] ConfidenceUpperBound { get; set; }
}