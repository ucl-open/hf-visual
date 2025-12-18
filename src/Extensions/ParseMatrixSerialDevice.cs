using System;
using System.Reactive.Linq;
using Bonsai;
using Bonsai.Harp;
using UclOpenHfVisualDataSchema;

public class ParseMatrixSerialDevice : Transform<string, Timestamped<MatrixArduinoData>>
{
    public override IObservable<Timestamped<MatrixArduinoData>> Process(IObservable<string> source)
    {
        return source.Select(value =>
        {
            var values = value.Split(',');
            var matrixArduinoData = new MatrixArduinoData
            {
                EncoderCount = Convert.ToInt32(values[0]),
                LickCountLeft = Convert.ToInt32(values[1]),
                LickCountRight = Convert.ToInt32(values[2]),
                LastSyncPulseTime = Convert.ToInt32(values[3]),
                PhotodiodeVal = Convert.ToInt32(values[4]),
                CurrentMs = Convert.ToInt32(values[5]),
            };
            return new Timestamped<MatrixArduinoData>(matrixArduinoData, matrixArduinoData.CurrentMs / 1000d);
        });
    }
}