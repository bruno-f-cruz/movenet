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
    public class BodyPart
    {
        public string Name;
        public Point2f Position;
        public float Confidence;
    }

    public class Pose : KeyedCollection<string, BodyPart>
    {
        public string[] BodyPartLabels = {
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"};

        public Pose(IplImage image)
        {
            Image = image;
        }

        public IplImage Image { get; }

        protected override string GetKeyForItem(BodyPart item)
        {
            return item.Name;
        }
    }
}
