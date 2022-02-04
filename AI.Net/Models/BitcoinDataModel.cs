using Microsoft.ML.Data;

namespace AI.Net.Models;

public class BitcoinDataModel
{
    [LoadColumn(0)]
    public string OpenTime { get; set; }

    [LoadColumn(1)]
    public float Open { get; set; }

    [LoadColumn(2)]
    public float High { get; set; }

    [LoadColumn(3)]
    public float Low { get; set; }

    [LoadColumn(4)]
    public float Close { get; set; }

    [LoadColumn(5)]
    public float BaseVolume { get; set; }

    [LoadColumn(6)]
    public string CloseTime { get; set; }

    [LoadColumn(7)]
    public float QuoteVolume { get; set; }

    [LoadColumn(8)]
    public int TradeCount { get; set; }

    [LoadColumn(9)]
    public float TakerBuyBaseVolume { get; set; }

    [LoadColumn(10)]
    public float TakerBuyQuoteVolume { get; set; }
}