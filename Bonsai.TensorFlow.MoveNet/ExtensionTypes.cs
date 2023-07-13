using OpenCV.Net;
using System.Collections.ObjectModel;

namespace Bonsai.TensorFlow.MoveNet
{
    /// <summary>
    /// Represents a body part, or node in the skeleton graph of the subject.
    /// </summary>
    public class BodyPart
    {
        /// <summary>
        /// Gets or sets the name of the body part.
        /// </summary>
        public string Name;

        /// <summary>
        /// Gets or sets the predicted location of the body part.
        /// </summary>
        public Point2f Position;

        /// <summary>
        /// Gets or sets the confidence score for the predicted location.
        /// </summary>
        public float Confidence;
    }

    /// <summary>
    /// Represents the result of pose estimation as a collection of body parts.
    /// </summary>
    public class Pose : KeyedCollection<string, BodyPart>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Pose"/> class
        /// extracted from the specified image.
        /// </summary>
        /// <param name="image">The image from which the pose was extracted.</param>
        public Pose(IplImage image)
        {
            Image = image;
        }

        /// <summary>
        /// Gets the image from which the pose was extracted.
        /// </summary>
        public IplImage Image { get; }

        /// <inheritdoc/>
        protected override string GetKeyForItem(BodyPart item)
        {
            return item.Name;
        }

        /// <summary>
        /// List of available bodyParts in the MoveNet network model.
        /// </summary>
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
    }
}
