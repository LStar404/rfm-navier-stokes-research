// CPU regression helper. It never launches a solver or calls CUDA.
using System;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.Web.Script.Serialization;

public static class ReaderFixture {
    static readonly JavaScriptSerializer Json=new JavaScriptSerializer();
    static void Write(string path,object value) { File.WriteAllText(path,Json.Serialize(value),Encoding.UTF8); }
    public static int Main(string[] args) {
        string mode=args[0], target=args[1], ready=args[2], stop=args[3], report=args[4];
        long start=Stopwatch.GetTimestamp(); int reads=0,errors=0; string lastError=null;
        Func<bool> continueReading=()=>!File.Exists(stop) && (Stopwatch.GetTimestamp()-start)/(double)Stopwatch.Frequency<10;
        if(mode=="hold_default" || mode=="hold_share_delete") {
            using(var stream=mode=="hold_default" ? File.OpenRead(target) :
                new FileStream(target,FileMode.Open,FileAccess.Read,FileShare.ReadWrite|FileShare.Delete)) {
                Write(ready,new { mode=mode, pid=Process.GetCurrentProcess().Id });
                while(continueReading())Thread.Sleep(1);
            }
        } else {
            var assembly=Assembly.LoadFrom(args[5]);
            var method=assembly.GetType("RankFairOuterWatchdog").GetMethod("Read",BindingFlags.Static|BindingFlags.NonPublic);
            if(method==null)throw new InvalidOperationException("Frozen Read method missing");
            Write(ready,new { mode=mode, pid=Process.GetCurrentProcess().Id, exact_frozen_method=method.Name });
            while(continueReading()) {
                try { method.Invoke(null,new object[]{target}); reads++; }
                catch(TargetInvocationException error) { errors++;lastError=error.InnerException.GetType().Name+": "+error.InnerException.Message; }
            }
        }
        Write(report,new { mode=mode, reads=reads, read_errors=errors, last_error=lastError });
        return 0;
    }
}
