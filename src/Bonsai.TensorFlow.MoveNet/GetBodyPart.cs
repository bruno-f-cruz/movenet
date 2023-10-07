using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace Bonsai.TensorFlow.MoveNet
{
    /// <summary>
    /// Represents an operator that returns the body part with the specified
    /// name for each pose in the sequence.
    /// </summary>
    [Description("Returns the body part with the specified name for each pose in the sequence.")]
    public class GetBodyPart : Transform<Pose, BodyPart>
    {
        /// <summary>
        /// Gets or sets the name of the body part to locate in each pose object.
        /// </summary>
        [TypeConverter(typeof(BodyPartConverter))]
        [Description("The name of the body part to locate in each pose object.")]
        public string Name { get; set; }

        /// <summary>
        /// Returns the body part with the specified name for each pose in an
        /// observable sequence.
        /// </summary>
        /// <param name="source">The sequence of poses for which to locate the body part.</param>
        /// <returns>
        /// A sequence of <see cref="BodyPart"/> objects representing the location
        /// of the body part with the specified name.
        /// </returns>
        public override IObservable<BodyPart> Process(IObservable<Pose> source)
        {
            return source.Select(pose =>
            {
                return pose[Name];
            });
        }

        class BodyPartConverter : StringConverter
        {
            public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
            {
                return new StandardValuesCollection(ExtensionMethods.GetBodyParts());
            }

            public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
            {
                return true;
            }
        }
    }
}
