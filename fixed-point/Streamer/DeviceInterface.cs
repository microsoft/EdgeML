using System;
using System.Text;
using System.IO.Ports;

namespace Streamer
{
	public class DeviceInterface
	{
		// Note: The baud rate used here must match with the baud rate specified in the .ino file
		int baud = 115200;

		SerialPort port = null;

		/*
		 * Returns the Serial Port which is running the prediction code for the given algo and dataset, if any.
		 * Else, exits by throwing an exception.
		*/
		public DeviceInterface(string algo, string dataset)
		{
			// Handshake messages used to establish connection with the Arduino device
			string syncMsg = algo;
			string acknMsg = dataset;

			// Trying to connect through each serial port
			foreach (string portName in SerialPort.GetPortNames())
			{
				try
				{
					port = new SerialPort(portName, baud);
					port.Open();
					port.ReadTimeout = 500;

					// Flush all the data in the buffer before synchronizing
					if (port.BytesToRead != 0)
					{
						var numBytes = port.BytesToRead;
						byte[] dummy = new byte[numBytes];
						port.Read(dummy, 0, numBytes);
					}

					// Convert the message into an array of bytes in ASCII format
					// Write the bytes into the serial buffer
					byte[] syncMsgBytes = Encoding.ASCII.GetBytes(syncMsg);
					port.Write(syncMsgBytes, 0, syncMsgBytes.Length);

					// Check if the device acknowledges with a reply containing the acknMsg
					// The reply will contain a carriage return (\r) in the end. Hence, using Contains() instead of Equals().
					var reply = port.ReadLine();
					if (reply.Contains(acknMsg))
						return;
				}
				catch (Exception)
				{
					try { port.Close(); } catch (Exception e) { Console.WriteLine(e.StackTrace); }
				}
			}

			throw new Exception("Couldn't find the device!");
		}

		// While streaming to the device, each feature (float) is represented as ASCII characters.
		// featureLength represents the number of ASCII characters used.
		int featureLength = 8;

		/*
		 * Streams the feature vector to the device for prediction and returns the predicted class ID along with the time taken for prediction.
		 */
		public int PredictOnDevice(string features, out ulong predictionTime)
		{
			try
			{
				try
				{
					// Convert each feature to bytes and stream to the device
					foreach (string featureStr in features.Split(new string[] { ", " }, StringSplitOptions.None))
					{
						string feature = featureStr;
						if (feature.Length == 0)
							throw new Exception("No features present in the data point");

						feature = ProcessFeature(feature);

						// Write the feature to the serial buffer
						byte[] bytes = Encoding.ASCII.GetBytes(feature.ToString());
						port.Write(bytes, 0, bytes.Length);
					}
				}
				catch (Exception e) { Console.WriteLine(e.StackTrace); }

				while (port.BytesToRead == 0) ;

				// Note: Prediction code identifies class indices from 0.
				// Hence, add one to the predicted label.
				var output = port.ReadLine();
				int classID = int.Parse(output) + 1;

				output = port.ReadLine();
				predictionTime = ulong.Parse(output);

				return classID;
			}
			catch (Exception e)
			{
				Console.WriteLine(e.StackTrace);

				if (port != null)
					port.Close();
			}

			throw new Exception("Unable to perform prediction on the device");
		}

		private string ProcessFeature(string feature)
		{
			// If the feature is an integer, convert to float by adding a decimal point
			if (!feature.Contains("."))
				feature = feature + ".";

			// Trim the feature to featureLength or extend the feature by adding trailing zeroes
			if (feature.Length > featureLength)
				feature = feature.Substring(0, featureLength);
			else
				feature = feature.PadRight(featureLength, '0');

			return feature;
		}
	}
}
