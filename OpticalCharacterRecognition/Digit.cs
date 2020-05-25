using Microsoft.ML.Data;

namespace OpticalCharacterRecognition
{
    public class Digit
    {
        [VectorType(784)] 
        public float[] PixelValues { get; set; }
    }
}
