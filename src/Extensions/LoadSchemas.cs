using System;
using System.ComponentModel;
using System.IO;
using System.Reactive.Linq;
using Bonsai;
using UclOpenHfVisualDataSchema;

public class LoadSchemas : Source<UclOpenHfVisualTaskParameters>
{
    [Description("The relative or absolute path of the file to open for reading.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string Path {get; set;}

    public override IObservable<UclOpenHfVisualTaskParameters> Generate()
    {
        string file = File.ReadAllText(Path);
        UclOpenHfVisualTaskParameters taskParameters = Newtonsoft.Json.JsonConvert.DeserializeObject<UclOpenHfVisualTaskParameters>(file);

        return Observable.Return(taskParameters);
    }
}