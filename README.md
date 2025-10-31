# Animetimm Inference

This project provides inference code for using AnimeTimm models to tag anime images. The code supports various CAFormer-based models from the AnimeTimm series.

## Installation

To run this project, you need to have the following dependencies installed:

- .NET SDK (for running the C# code)
- Microsoft.ML.OnnxRuntime
- SixLabors.ImageSharp
- CsvHelper
- System.Text.Json

You can install these dependencies using NuGet package manager.

## Usage

1. **Prepare the Model and Data**:
   - Download the desired AniTimm model from Hugging Face.
   - Prepare your image data and tag metadata CSV file.

2. **Run Inference**:
   - Use the following command to run inference on an image:
     ```bash
     dotnet run -- <model.onnx> <image_path> <selected_tags.csv> <config.json>
     ```

3. **Example**:
   - For the largest model (`caformer_b36.dbv4-full`):
     ```bash
     dotnet run -- caformer_b36.dbv4-full.onnx image.jpg selected_tags.csv config.json
     ```

   - For smaller models (e.g., `caformer_s18.dbv4-full`):
     ```bash
     dotnet run -- caformer_s18.dbv4-full.onnx image.jpg selected_tags.csv config.json
     ```

## Models

The following CAFormer-based models from the AniTimm series are supported:

| Model Name                                  | Description                          |
|---------------------------------------------|--------------------------------------|
| [caformer_b36.dbv4-full](https://huggingface.co/animetimm/caformer_b36.dbv4-full) | Largest model with highest accuracy  |
| [caformer_s18.dbv4-full](https://huggingface.co/animetimm/caformer_s18.dbv4-full) | Smaller model with balanced performance |

## Output

The inference results will be displayed in the console, showing the top predicted tags for the input image, categorized into general tags, character tags, and rating tags.

## License

This project is licensed under the MIT License.
