using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using TensorFlow;
using System.Collections.ObjectModel;
using System.Reflection;
using System.IO;

namespace Bonsai.TensorFlow.MoveNet
{
    [Description("Performs human pose estim ation using a MoveNet network")]
    public class PredictSinglePoseLightning : Transform<IplImage, Pose>
    {
        private int InputSize = 192;

        public float MinimumConfidence { get; set; } = 0;

        private IObservable<Pose> Process(IObservable<IplImage[]> source)
        {
            return Observable.Defer(() =>
            {
                IplImage resizeTemp = null;
                TFTensor tensor = null;
                TFSession.Runner runner = null;

                var basePath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                const string ModelName = "movenet_singlepose_lightning_v4.pb";
                var defaultPath = Path.Combine(basePath, ModelName);

                if (!File.Exists(defaultPath)) defaultPath = Path.Combine(basePath, "..\\..\\content\\", ModelName);

                // check if pb is next to dll if not:
                var graph = TensorHelper.ImportModel(defaultPath, out TFSession session);
                
                // Expected input size
                var tensorSize = new Size(InputSize, InputSize);
                
                return source.Select(input =>
                {
                    int colorChannels = input[0].Channels;
                    var initialSize = input[0].Size;

                    var batchSize = input.Length;
                    if (batchSize > 1) { throw new NotImplementedException("Batch processing not implemented"); }

                    if (tensor == null || tensor.Shape[0] != batchSize || tensor.Shape[1] != tensorSize.Height || tensor.Shape[2] != tensorSize.Width )
                    {
                        tensor?.Dispose();
                        runner = session.GetRunner();
                        tensor = TensorHelper.CreatePlaceholder(graph, runner, tensorSize, batchSize, colorChannels, TFDataType.Int32);
                        runner.Fetch(graph["Identity"][0]);
                    }

                    var frames = Array.ConvertAll(input, frame => 
                    {
                        frame = TensorHelper.EnsureFrameSize(frame, tensorSize, ref resizeTemp);
                        return frame;
                    });
                    TensorHelper.UpdateTensor(tensor, colorChannels, Depth.S32, frames);
                    var output = runner.Run();
                    var out0 = output[0];
                    float[,,,] out0_arr = new float[out0.Shape[0], out0.Shape[1], out0.Shape[2], out0.Shape[3]];
                    out0.GetValue(out0_arr);

                    var pose = new Pose(input[0]);

                    for (int i = 0; i < out0.Shape[2]; i++)
                    {
                        var part = new BodyPart();
                        part.Confidence = out0_arr[0, 0, i, 2];
                        if (part.Confidence > MinimumConfidence)
                        {
                            part.Position.X = out0_arr[0, 0, i, 1] * (float)initialSize.Width;
                            part.Position.Y = out0_arr[0, 0, i, 0] * (float)initialSize.Height;
                        }
                        else
                        {
                            part.Position.X = float.NaN;
                            part.Position.Y = float.NaN;
                        }
                        part.Name = pose.BodyPartLabels[i];
                        pose.Add(part);
                    }
                    return pose;
                });
            });
        }

        public override IObservable<Pose> Process(IObservable<IplImage> source)
        {
            return Process(source.Select(frame => new IplImage[] { frame }));
        }
    }
}


