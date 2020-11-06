// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.IO;

namespace Streamer
{
	public enum DataType { Float, Int16, Int32 };

	class Program
	{
		DataType datatype = DataType.Float;

		static int Main(string[] args)
		{
			new Program().Run();
			return 0;
		}

		string[] X, Y;
		string outputFile;

		public void Run()
		{
			if (datatype == DataType.Float)
				Console.WriteLine("Reading data in FLOAT format");
			else if (datatype == DataType.Int16)
				Console.WriteLine("Reading data in INT16 format");
			else
				Console.WriteLine("Reading data in INT32 format");

			string projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
			string inputDir = Path.Combine(projectDir, "input");

			string outputDir = "output";
			Directory.CreateDirectory(Path.Combine(projectDir, outputDir));

			outputFile = Path.Combine(projectDir, outputDir, "prediction-info.txt");

			ReadDataset(Path.Combine(inputDir, "X.csv"), Path.Combine(inputDir, "Y.csv"));

			DeviceInterface device = new DeviceInterface("fixed", "point", datatype);

			PerformPrediction(device);

			return;
		}

		// Read and validate the dataset
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

			using (StreamWriter file = new StreamWriter(outputFile))
			{
				// Read each data point, predict on device and compare the class ID
				for (int i = 0; i < X.Length; i++)
				{
					int classID = device.PredictOnDevice(X[i], out ulong predictionTime);
					var label = Y[i];

					if (classID.ToString().Equals(label))
					{
						Console.WriteLine((i + 1) + ": Correct prediction in " + predictionTime + " \u00b5sec");
						correct++;
					}
					else
					{
						Console.WriteLine((i + 1) + ": Incorrect prediction" + classID + "/" + label);
						//file.WriteLine("Incorrect prediction for input " + (i + 1));
						file.WriteLine("Incorrect prediction for input " + (total + 1) + ". Predicted " + classID + " Expected " + label);
					}

					totalPredictionTime += predictionTime;
					total++;
					Console.WriteLine("Accuracy: " + (((float)correct / total) * 100));
				}

				file.WriteLine("\n\n#test points = " + total);
				file.WriteLine("Correct predictions = " + correct);
				file.WriteLine("Accuracy = " + (((float)correct / total) * 100).ToString("0.000") + "\n");

				Console.WriteLine("\n\nCorrect: " + correct);
				Console.WriteLine("Accuracy: " + (((float)correct / total) * 100));
				Console.WriteLine("Average prediction time: " + ((float)totalPredictionTime / total) + " \u00b5sec\n");
			}

			return;
		}
	}
}
