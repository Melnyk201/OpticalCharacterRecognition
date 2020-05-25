using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace OpticalCharacterRecognition
{
    class Program
    {
        private const string DATA_PATH = @"..\..\..\handwritten_digits_large.csv";
        static void Main(string[] args)
        {
            var context = new MLContext();

            Console.WriteLine("Loading data....");
            var dataView = context.Data.LoadFromTextFile(
                path: DATA_PATH,
                columns: new[]
                {
                    new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                    new TextLoader.Column("Number", DataKind.Single, 0)
                },
                hasHeader: false,
                separatorChar: ',');

            var partitions = context.MulticlassClassification.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate(
                DefaultColumnNames.Features,
                nameof(Digit.PixelValues))
                      .AppendCacheCheckpoint(context)
                      .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    labelColumnName: "Number",
                    featureColumnName: DefaultColumnNames.Features));

            Console.WriteLine("Training model....");
            var model = pipeline.Fit(partitions.TrainSet);
            Console.WriteLine("Evaluating model....");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions,
                label: "Number",
                score: DefaultColumnNames.Score);
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.AccuracyMicro:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.AccuracyMacro:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            // grab three digits from the data: 2, 7, and 3
            var digits = context.Data.CreateEnumerable<Digit>(dataView, reuseRowObject: false).ToArray();
            var testDigits = new Digit[] { digits[5], digits[12], digits[22] };
            var engine = model.CreatePredictionEngine<Digit, DigitPrediction>(context);
            for (var i = 0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);
                // show results
                Console.WriteLine($"Predicting test digit {i}...");
                for (var j = 0; j < 10; j++)
                {
                    Console.WriteLine($"  {j}: {prediction.Score[j]:P2}");
                }
                Console.WriteLine();
            }
            Console.ReadKey();
        }
    }
}
