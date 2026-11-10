// CPU-only immutable-name publication proof; not a replacement watchdog.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
public static class ImmutableReaderFixture {
    public static int Main(string[] args) {
        var json=new JavaScriptSerializer(); string root=args[0];
        var method=Assembly.LoadFrom(args[1]).GetType("RankFairOuterWatchdog").GetMethod("Read",BindingFlags.Static|BindingFlags.NonPublic);
        int consumed=0; string error=null; long start=Stopwatch.GetTimestamp();
        File.WriteAllText(Path.Combine(root,"reader_ready.json"),"{}",Encoding.UTF8);
        try {
            for(int sequence=0;sequence<512;sequence++) {
                string path=Path.Combine(root,"state_"+sequence.ToString("D4")+".json");
                while(!File.Exists(path)) {
                    if((Stopwatch.GetTimestamp()-start)/(double)Stopwatch.Frequency>15)throw new TimeoutException("Missing immutable sequence");
                    Thread.Sleep(1);
                }
                var value=(Dictionary<string,object>)method.Invoke(null,new object[]{path});
                string phase=sequence%2==0 ? "active" : "terminal";
                if(value.Count!=4 || Convert.ToInt32(value["sequence"])!=sequence || Convert.ToInt32(value["index"])!=sequence/2 ||
                    (string)value["phase"]!=phase || (string)value["payload"]!=(sequence.ToString()+":"+new string('x',4096)))
                    throw new InvalidDataException("Torn JSON or incorrect sequence");
                consumed++;
            }
        } catch(Exception failure) { error=failure.ToString(); }
        File.WriteAllText(Path.Combine(root,"reader_result.json"),json.Serialize(new { consumed=consumed, errors=error==null?0:1,
            error=error, exact_frozen_read=true, wall_seconds=(Stopwatch.GetTimestamp()-start)/(double)Stopwatch.Frequency }),Encoding.UTF8);
        return error==null ? 0 : 1;
    }
}
