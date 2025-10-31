using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Advanced;
using CsvHelper;
using System.Text.Json;

class TimmConfig
{
    public string[] tags { get; set; } = [];
}
class TagRecord
{
    public string name { get; set; } = string.Empty;
    public int category { get; set; }
    public float best_threshold { get; set; }
}

class CaformerDbv4Inference
{
    static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    static readonly float[] Std = [0.229f, 0.224f, 0.225f];
    const int PadSize = 512;
    const int FinalSize = 384;

    static DenseTensor<float> Preprocess(string imagePath)
    {
        using var image = Image.Load<Rgba32>(imagePath);

        int targetSize = Math.Max(Math.Max(image.Width, image.Height), PadSize);
        using var padded = new Image<Rgba32>(targetSize, targetSize, new Rgba32(255, 255, 255, 255));

        int offsetX = (targetSize - image.Width) / 2;
        int offsetY = (targetSize - image.Height) / 2;

        padded.Mutate(x => x.DrawImage(image, new Point(offsetX, offsetY), 1.0f));
        padded.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(FinalSize, FinalSize),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic
        }));

        Rectangle cropRect = new(
            (padded.Width - FinalSize) / 2, (padded.Height - FinalSize) / 2,
            FinalSize, FinalSize
        );
        padded.Mutate(x => x.Crop(cropRect));

        var tensor = new DenseTensor<float>([1, 3, FinalSize, FinalSize]); // BCHW
        for (int y = 0; y < FinalSize; y++)
        {
            Span<Rgba32> rowSpan = padded.DangerousGetPixelRowMemory(y).Span;
            for (int x = 0; x < FinalSize; x++)
            {
                var px = rowSpan[x];
                float r = px.R / 255f; float g = px.G / 255f; float b = px.B / 255f;
                tensor[0, 0, y, x] = (r - Mean[0]) / Std[0]; tensor[0, 1, y, x] = (g - Mean[1]) / Std[1]; tensor[0, 2, y, x] = (b - Mean[2]) / Std[2];
            }
        }
        return tensor;
    }

    static List<TagRecord> LoadTagMeta(string csvPath)
    {
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        return [.. csv.GetRecords<TagRecord>()];
    }

    static void Main(string[] args)
    {
        if (args.Length == 1)
            ShowInferenceResults(args[0], Inference("model.onnx", args[0], "selected_tags.csv", "config.json"));
        else if (args.Length < 4)
            Console.WriteLine("Usage: dotnet run -- <model.onnx> <image_path> <selected_tags.csv> <config.json>");
        else
            ShowInferenceResults(args[1], Inference(args[0], args[1], args[2], args[3]));
    }

    static void ShowInferenceResults(string imagePath, (List<(float, string)> general_tags, List<(float, string)> character_tags, List<(float, string)> rating_tags) results)
    {
        List<(float, string)> general_tags, character_tags, rating_tags;
        (general_tags, character_tags, rating_tags) = results;
        Console.WriteLine($"Tagged results for {Path.GetFileName(imagePath)}:");
        Console.WriteLine(" General Tags:");
        ShowResult(general_tags);
        Console.WriteLine(" Character Tags:");
        ShowResult(character_tags);
        Console.WriteLine(" Rating Tags:");
        ShowResult(rating_tags);
    }

    static void ShowResult(List<(float, string)> general_tags)
    {
        foreach (var (prob, label) in general_tags)
            Console.WriteLine($"  - {label} ({prob:P2})");
    }

    static (List<(float, string)> general_tags, List<(float, string)> character_tags, List<(float, string)> rating_tags)
        Inference(string modelPath, string imagePath, string tagsCsv, string configPath)
    {
        var inputTensor = Preprocess(imagePath);
        var tagMeta = LoadTagMeta(tagsCsv).ToDictionary(x => x.name);
        var timmConfig = JsonSerializer.Deserialize<TimmConfig>(File.ReadAllText(configPath))!;
        var outputMap = timmConfig.tags;
        using var session = new InferenceSession(modelPath);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        using var results = session.Run(inputs);
        float[] probs = [.. results.First(x => x.Name == "prediction").AsEnumerable<float>()];
        var probs_with_label = probs.Zip(outputMap).OrderByDescending(x => x.First).ToList();
        return (
            probs_with_label.Where(x => tagMeta[x.Second].category == 0 && x.First > tagMeta[x.Second].best_threshold).Take(50).ToList(),
            probs_with_label.Where(x => tagMeta[x.Second].category == 4 && x.First > tagMeta[x.Second].best_threshold).Take(30).ToList(),
            probs_with_label.Where(x => tagMeta[x.Second].category == 9).ToList()
        );
    }
}