using System.IO;
using System.Reflection;

namespace Bonsai.TensorFlow.MoveNet
{
    internal static class ResourceHelper
    {
        public static string FindResourcePath(string fileName)
        {
            var basePath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var defaultPath = Path.Combine(basePath, fileName);
            return !File.Exists(defaultPath)
                ? Path.Combine(basePath, "..\\..\\content\\", fileName)
                : defaultPath;
        }
    }
}
