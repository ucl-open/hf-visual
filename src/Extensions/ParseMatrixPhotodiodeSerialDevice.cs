using System;
using System.Reactive.Linq;
using Bonsai;
using Bonsai.Harp;
using UclOpenHfVisualDataSchema;

public class ParseMatrixPhotodiodeSerialDevice : Transform<string, MatrixArduinoPhotodiodeData>
{
    public override IObservable<MatrixArduinoPhotodiodeData> Process(IObservable<string> source)
    {
        return source.Select(value =>
        {
            var values = value.Split(',');
            var matrixArduinoData = new MatrixArduinoPhotodiodeData
            {
                PhotodiodeVal = Convert.ToInt32(values[0]),
                SyncVal = Convert.ToInt32(values[0])
            };
            return matrixArduinoData;
        });
    }
}