using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using TensorFlow;

namespace Bonsai.TensorFlow.MoveNet
{
    /// <summary>
    /// Represents an operator that performs pose estimation on an 256x256 RGB image
    /// by running inference on a "movenet_singlepose_thunder_v4" network. The network
    /// will identify a single skeleton in the image.
    /// </summary>
    /// <seealso cref="PredictSinglePoseLightning"/>
    /// <seealso cref="PredictMultiPoseLightning"/>
    [Description("Performs human pose estimation using a MoveNet network")]
    public class PredictSinglePoseThunder : Transform<IplImage, Pose>
    {

        /// <summary>
        /// Expected input image size.
        /// </summary>
        const int InputSize = 256;

        /// <summary>
        /// Gets or sets a value specifying the confidence threshold used to discard predicted
        /// body part positions. If no value is specified, all estimated positions are returned.
        /// </summary>
        [Range(0, 1)]
        [Editor(DesignTypes.SliderEditor, DesignTypes.UITypeEditor)]
        [Description("Specifies the confidence threshold used to discard predicted body part positions. If no value is specified, all estimated positions are returned.")]
        public float MinimumConfidence { get; set; } = 0;

        /// <summary>
        /// Gets or sets the optional color conversion used to prepare images for inference.
        /// </summary>
        [Description("The optional color conversion used to prepare images for inference.")]
        public ColorConversion? ColorConversion { get; set; } = OpenCV.Net.ColorConversion.Bgr2Rgb;

        /// <summary>
        /// Performs markerless, single instance, human pose estimation for each array
        /// of images in an observable sequence using a "movenet_singlepose_thunder_v4" model.
        /// </summary>
        /// <param name="source">The sequence of image batches from which to extract the poses.</param>
        /// <returns>
        /// A sequence of <see cref="Pose"/> objects representing the results
        /// of pose estimation for each image batch in the <paramref name="source"/>
        /// sequence.
        /// </returns>
        private IObservable<Pose> Process(IObservable<IplImage[]> source)
        {
            return Observable.Defer(() =>
            {
                IplImage resizeTemp = null;
                IplImage colorTemp = null;
                TFTensor tensor = null;
                TFSession.Runner runner = null;
                var availableBodyParts = ExtensionMethods.GetBodyParts();
                var modelPath = ResourceHelper.FindResourcePath("movenet_singlepose_thunder_v4.pb");
                var graph = TensorHelper.ImportModel(modelPath, out TFSession session);

                var tensorSize = new Size(InputSize, InputSize);
                return source.Select(input =>
                {
                    int colorChannels = input[0].Channels;
                    var initialSize = input[0].Size;

                    var batchSize = input.Length;
                    if (batchSize > 1) { throw new NotImplementedException("Batch processing not implemented"); }

                    if (tensor == null || tensor.Shape[0] != batchSize || tensor.Shape[1] != tensorSize.Height || tensor.Shape[2] != tensorSize.Width)
                    {
                        tensor?.Dispose();
                        runner = session.GetRunner();
                        tensor = TensorHelper.CreatePlaceholder(graph, runner, tensorSize, batchSize, colorChannels, TFDataType.Int32);
                        runner.Fetch(graph["Identity"][0]);
                    }

                    var frames = Array.ConvertAll(input, frame =>
                    {
                        frame = TensorHelper.EnsureFrameSize(frame, tensorSize, ref resizeTemp);
                        frame = TensorHelper.EnsureColorFormat(frame, ColorConversion, ref colorTemp);
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
                            part.Position.X = out0_arr[0, 0, i, 1] * initialSize.Width;
                            part.Position.Y = out0_arr[0, 0, i, 0] * initialSize.Height;
                        }
                        else
                        {
                            part.Position.X = float.NaN;
                            part.Position.Y = float.NaN;
                        }
                        part.Name = availableBodyParts[i];
                        pose.Add(part);
                    }
                    return pose;
                });
            });
        }

        /// <summary>
        /// Performs markerless, single instance, human pose estimation for each
        /// image in an observable sequence using a "movenet_singlepose_thunder_v4" model.
        /// </summary>
        /// <param name="source">The sequence of image batches from which to extract the poses.</param>
        /// <returns>
        /// A sequence of <see cref="Pose"/> objects representing the results
        /// of pose estimation for each image in the <paramref name="source"/>
        /// sequence.
        /// </returns>
        public override IObservable<Pose> Process(IObservable<IplImage> source)
        {
            return Process(source.Select(frame => new IplImage[] { frame }));
        }
    }
}


