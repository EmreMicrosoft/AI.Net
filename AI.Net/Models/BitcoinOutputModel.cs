﻿namespace AI.Net.Models;

public class BitcoinOutputModel
{
    public float[] PredictedValues { get; set; }
    public float[] LowerBoundValues { get; set; }
    public float[] UpperBoundValues { get; set; }
}