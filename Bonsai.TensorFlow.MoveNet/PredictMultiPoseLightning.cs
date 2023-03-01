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
    public class PredictMultiPoseLightning : Transform<IplImage, Pose[]>
    {
        public int InputSize { get; set; } = 256;

        public float MinimumConfidence { get; set; } = 0;

        private IObservable<Pose[]> Process(IObservable<IplImage[]> source)
        {
            return Observable.Defer(() =>
            {
                IplImage resizeTemp = null;
                TFTensor tensor = null;
                TFSession.Runner runner = null;

                var basePath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                const string ModelName = "movenet_multipose_lightning_v1.pb";
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
                    float[,,] out0_arr = new float[out0.Shape[0], out0.Shape[1], out0.Shape[2]];
                    out0.GetValue(out0_arr);

                    const int batchIdx = 0;
                    const int nBodyPart = 13;
                    var poseCollection = new Pose[6]; // The output of the network seems to always be 1x6x56
                    for (int j = 0; j < 6; j++)
                    {
                        var pose = new Pose(input[0]);

                        for (int i = 0; i < nBodyPart; i++)
                        {
                            var part = new BodyPart();
                            part.Confidence = out0_arr[batchIdx, j, (i*3) + 2];
                            if (part.Confidence > MinimumConfidence)
                            {
                                part.Position.X = out0_arr[batchIdx, j, (i * 3) + 1] * (float)initialSize.Width;
                                part.Position.Y = out0_arr[batchIdx, j, (i * 3) + 0] * (float)initialSize.Height;
                            }
                            else
                            {
                                part.Position.X = float.NaN;
                                part.Position.Y = float.NaN;
                            }
                            part.Name = pose.BodyPartLabels[i];
                            pose.Add(part);
                        }
                        poseCollection[j] = pose;
                    }
                    return poseCollection;
                });
            });
        }

        public override IObservable<Pose[]> Process(IObservable<IplImage> source)
        {
            return Process(source.Select(frame => new IplImage[] { frame }));
        }
    }
}


