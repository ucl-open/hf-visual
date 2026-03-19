using System;
using System.ComponentModel;
using System.IO;
using System.Reactive.Linq;
using Bonsai;
using UclOpenHfVisualDataSchema;

public class LoadSchemas : Source<Tuple<UclOpenSession, UclOpenHfVisualRig, UclOpenHfVisualTaskLogic>>
{
    [Description("The relative or absolute path of the session file to open for reading.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string SessionSettingsPath {get; set;}

    [Description("The relative or absolute path of the rig file to open for reading.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string RigSettingsPath {get; set;}

    [Description("The relative or absolute path of the task file to open for reading.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string TaskLogicSettingsPath {get; set;}

    public override IObservable<Tuple<UclOpenSession, UclOpenHfVisualRig, UclOpenHfVisualTaskLogic>> Generate()
    {
        string sessionSettingsRaw = File.ReadAllText(SessionSettingsPath);
        string rigSettingsRaw = File.ReadAllText(RigSettingsPath);
        string taskLogicSettingsRaw = File.ReadAllText(TaskLogicSettingsPath);

        var sessionSource = Observable.Return(Newtonsoft.Json.JsonConvert.DeserializeObject<UclOpenSession>(sessionSettingsRaw));
        var rigSource = Observable.Return(Newtonsoft.Json.JsonConvert.DeserializeObject<UclOpenHfVisualRig>(rigSettingsRaw));
        var taskLogicSource = Observable.Return(Newtonsoft.Json.JsonConvert.DeserializeObject<UclOpenHfVisualTaskLogic>(taskLogicSettingsRaw));

        return sessionSource.Zip(rigSource, taskLogicSource, (s1, s2, s3) =>
        {
            return Tuple.Create(s1, s2, s3);
        });
    }
}