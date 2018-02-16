using System;
using System.IO;

namespace Streamer
{
	class Program
	{
		int limit = -1; // -1 to stream the entire dataset

		static int Main(string[] args)
		{
			new Program().Run();
			return 0;
		}

		string[] X, Y;
		string outputFile;

		public void Run()
		{
			string projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
			string inputDir = Path.Combine(projectDir, "input");

			string outputDir = "output";
			Directory.CreateDirectory(Path.Combine(projectDir, outputDir));

			outputFile = Path.Combine(projectDir, outputDir, "prediction-info.txt");

			ReadDataset(Path.Combine(inputDir, "X.csv"), Path.Combine(inputDir, "Y.csv"));

			DeviceInterface device = new DeviceInterface("fixed", "point");

			PerformPrediction(device);

			return;
		}

		public void ReadDataset(string X_file, string Y_file)
		{
			X = File.ReadAllLines(X_file);
			Y = File.ReadAllLines(Y_file);

			if (X.Length != Y.Length)
				throw new Exception("Number of data points not equal to the number of labels");

			int featuresLength = X[0].Split(new string[] { ", " }, StringSplitOptions.None).Length;
			int labelsLength = 1;

			// Validating the dataset
			for (int i = 0; i < X.Length; i++)
			{
				int X_length = X[i].Split(new string[] { ", " }, StringSplitOptions.None).Length;
				int Y_length = Y[i].Split(new string[] { ", " }, StringSplitOptions.None).Length;

				if (X_length != featuresLength || Y_length != labelsLength)
					throw new Exception("Number of features or number of labels not consistent");

				Y[i] = Y[i].Split(new string[] { ", " }, StringSplitOptions.None)[0];
			}

			return;
		}

		public void PerformPrediction(DeviceInterface device)
		{
			int correct = 0, total = 0;
			ulong totalPredictionTime = 0;

			int ite = limit;
			if (limit == -1)
				ite = X.Length;

			using (StreamWriter file = new StreamWriter(outputFile))
			{
				for (int i = 0; i < ite; i++)
				{
					int classID = device.PredictOnDevice(X[i], out ulong predictionTime);

					if (classID.ToString().Equals(Y[i]))
					{
						Console.WriteLine((i + 1) + ": Correct prediction in " + predictionTime + " \u00b5sec");
						correct++;
					}
					else
					{
						Console.WriteLine((i + 1) + ": Incorrect prediction");
						file.WriteLine("Incorrect prediction for input " + (i + 1));
					}

					totalPredictionTime += predictionTime;
					total++;
				}

				file.WriteLine("\n\nCorrect: " + correct);
				file.WriteLine("Accuracy: " + (((float)correct / total) * 100));
				file.WriteLine("Average prediction time: " + ((float)totalPredictionTime / total) + " \u00b5sec\n");

				Console.WriteLine("\n\nCorrect: " + correct);
				Console.WriteLine("Accuracy: " + (((float)correct / total) * 100));
				Console.WriteLine("Average prediction time: " + ((float)totalPredictionTime / total) + " \u00b5sec\n");
			}

			return;
		}
	}
}
