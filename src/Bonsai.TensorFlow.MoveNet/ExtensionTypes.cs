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
    }
}
