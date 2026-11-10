// Separate non-Python observer and Job owner. No termination API runs in observer.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

public static class RankFairOuterWatchdog {
    static readonly JavaScriptSerializer Json = new JavaScriptSerializer();
    const uint WAIT_OBJECT_0=0, WAIT_TIMEOUT=258, TERMINATION_CODE=772;
    const uint CREATE_SUSPENDED=4, CREATE_NO_WINDOW=0x08000000, JOB_KILL_ON_CLOSE=0x2000;
    [StructLayout(LayoutKind.Sequential)] struct SECURITY_ATTRIBUTES { public int length; public IntPtr descriptor; public int inherit; }
    [StructLayout(LayoutKind.Sequential,CharSet=CharSet.Unicode)] struct STARTUPINFO {
        public int cb; public string reserved,desktop,title; public uint x,y,xSize,ySize,xChars,yChars,fill,flags;
        public short show,reserved2; public IntPtr reservedPointer,input,output,error;
    }
    [StructLayout(LayoutKind.Sequential)] struct PROCESS_INFORMATION { public IntPtr process,thread; public uint pid,tid; }
    [StructLayout(LayoutKind.Sequential)] struct STARTUPINFOEX { public STARTUPINFO startup; public IntPtr attributes; }
    [StructLayout(LayoutKind.Sequential)] struct IO_COUNTERS { public ulong read,write,other,readBytes,writeBytes,otherBytes; }
    [StructLayout(LayoutKind.Sequential)] struct BASIC_LIMIT {
        public long processTime,jobTime; public uint flags; public UIntPtr minimum,maximum; public uint active; public UIntPtr affinity; public uint priority,scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct EXTENDED_LIMIT {
        public BASIC_LIMIT basic; public IO_COUNTERS io; public UIntPtr processMemory,jobMemory,peakProcess,peakJob;
    }
    [StructLayout(LayoutKind.Sequential)] struct BASIC_ACCOUNTING {
        public long user,kernel,periodUser,periodKernel; public uint faults,totalProcesses,activeProcesses,terminatedProcesses;
    }
    [DllImport("kernel32",SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr security,string name);
    [DllImport("kernel32",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job,int kind,ref EXTENDED_LIMIT value,uint length);
    [DllImport("kernel32",SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job,int kind,out BASIC_ACCOUNTING value,uint length,IntPtr returned);
    [DllImport("kernel32",SetLastError=true)] static extern bool IsProcessInJob(IntPtr process,IntPtr job,out bool answer);
    [DllImport("kernel32",SetLastError=true)] static extern bool InitializeProcThreadAttributeList(IntPtr list,int count,int flags,ref IntPtr size);
    [DllImport("kernel32",SetLastError=true)] static extern bool UpdateProcThreadAttribute(IntPtr list,uint flags,IntPtr attribute,IntPtr value,IntPtr size,IntPtr previous,IntPtr returned);
    [DllImport("kernel32")] static extern void DeleteProcThreadAttributeList(IntPtr list);
    [DllImport("kernel32",SetLastError=true)] static extern bool TerminateJobObject(IntPtr job,uint code);
    [DllImport("kernel32",SetLastError=true)] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32",SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle,uint ms);
    [DllImport("kernel32",SetLastError=true)] static extern uint ResumeThread(IntPtr handle);
    [DllImport("kernel32",SetLastError=true)] static extern bool GetProcessTimes(IntPtr handle,out long born,out long exit,out long kernel,out long user);
    [DllImport("kernel32",SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr handle,out uint code);
    [DllImport("kernel32")] static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32",SetLastError=true)] static extern bool SetProcessAffinityMask(IntPtr process,UIntPtr mask);
    [DllImport("kernel32",SetLastError=true)] static extern bool GetProcessAffinityMask(IntPtr process,out UIntPtr mask,out UIntPtr system);
    [DllImport("kernel32",SetLastError=true,CharSet=CharSet.Unicode)] static extern bool CreateProcess(
        string application,StringBuilder command,IntPtr processSecurity,IntPtr threadSecurity,bool inherit,
        uint flags,IntPtr environment,string directory,ref STARTUPINFOEX startup,out PROCESS_INFORMATION process);

    static Dictionary<string,object> D(params object[] pairs) {
        var value=new Dictionary<string,object>();
        for(int i=0;i<pairs.Length;i+=2)value.Add((string)pairs[i],pairs[i+1]);
        return value;
    }
    static void Need(bool ok,string message) { if(!ok)throw new InvalidOperationException(message+"; Win32="+Marshal.GetLastWin32Error()); }
    static long Stamp() { return Stopwatch.GetTimestamp(); }
    static double Seconds(long start) { return (Stamp()-start)/(double)Stopwatch.Frequency; }
    static long Nanoseconds() { long t=Stamp(),f=Stopwatch.Frequency; return (t/f)*1000000000L+((t%f)*1000000000L)/f; }
    static Dictionary<string,object> Read(string path) { return Json.Deserialize<Dictionary<string,object>>(File.ReadAllText(path)); }
    static void Write(string path,object value) {
        byte[] bytes=Encoding.UTF8.GetBytes(Json.Serialize(value)+"\n");
        using(var stream=new FileStream(path+".pending",FileMode.CreateNew,FileAccess.Write,FileShare.Read)) {
            stream.Write(bytes,0,bytes.Length); stream.Flush(true);
        }
        File.Move(path+".pending",path);
    }
    static Dictionary<string,object> Identity(IntPtr handle,uint pid) {
        long b,e,k,u; Need(GetProcessTimes(handle,out b,out e,out k,out u),"GetProcessTimes");
        return D("pid",pid,"birth_filetime_100ns",b,"exit_filetime_100ns",e,"kernel_cpu_100ns",k,"user_cpu_100ns",u);
    }
    static string Quote(string value) {
        var output=new StringBuilder("\""); int slashes=0;
        foreach(char c in value) {
            if(c=='\\') { slashes++; continue; }
            if(c=='\"') { output.Append('\\',2*slashes+1); output.Append(c); }
            else { output.Append('\\',slashes); output.Append(c); }
            slashes=0;
        }
        output.Append('\\',2*slashes); return output.Append('"').ToString();
    }
    static uint Active(IntPtr job) {
        BASIC_ACCOUNTING counters;
        Need(QueryInformationJobObject(job,1,out counters,(uint)Marshal.SizeOf(typeof(BASIC_ACCOUNTING)),IntPtr.Zero),"QueryInformationJobObject");
        return counters.activeProcesses;
    }
    static string Self() { return Assembly.GetExecutingAssembly().Location; }
    static string Hash(string path) {
        using(var stream=File.OpenRead(path))using(var hash=SHA256.Create())return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant();
    }
    static void CheckInputs(Dictionary<string,object> request) {
        Need(request.ContainsKey("inputs_sha256"),"Input hash bindings required");
        var inputs=(Dictionary<string,object>)request["inputs_sha256"];
        Need(inputs.Count>0,"Nonempty input guard required");
        foreach(var pair in inputs)Need(Hash(pair.Key)==(string)pair.Value,"Input changed: "+Path.GetFileName(pair.Key));
    }
    static ulong Pin(Dictionary<string,object> request,string field,int defaultCpu) {
        int cpu=request.ContainsKey(field) ? Convert.ToInt32(request[field]) : defaultCpu;
        Need(cpu>=0 && cpu<64,"Affinity CPU rejected"); ulong mask=1UL<<cpu;
        Need(SetProcessAffinityMask(GetCurrentProcess(),new UIntPtr(mask)),"SetProcessAffinityMask");
        UIntPtr actual,system; Need(GetProcessAffinityMask(GetCurrentProcess(),out actual,out system) && actual.ToUInt64()==mask,"Affinity check");
        return mask;
    }
    sealed class SlotGuard {
        readonly Dictionary<string,object> request; readonly string directory;
        int lastIndex=-1; long lastLaunch=0,lastDeadline=0; string lastPhase=null,lastBytes=null;
        public Dictionary<string,object> latest=null;
        public SlotGuard(Dictionary<string,object> r,string d) { request=r; directory=d; }
        public string Poll() {
            if(!request.ContainsKey("slot_state_path"))return null;
            try {
                string path=(string)request["slot_state_path"];
                var state=TryRead(path);
                if(state==null)return lastPhase=="active" && Nanoseconds()>=lastDeadline ? "active_slot_deadline_exceeded" : null;
                var ready=TryRead(Path.Combine(directory,"owner_ready.json")); Need(ready!=null,"Slot preceded exact coordinator registration");
                var expected=(Dictionary<string,object>)ready["coordinator_identity"];
                var actual=(Dictionary<string,object>)state["coordinator"];
                Need(Convert.ToInt64(actual["pid"])==Convert.ToInt64(expected["pid"]) &&
                     Convert.ToInt64(actual["birth_filetime_100ns"])==Convert.ToInt64(expected["birth_filetime_100ns"]),"Slot coordinator identity differs");
                Need((string)state["schema"]=="rfm-v8-outer-slot-1" && (string)state["run_token"]==(string)request["run_token"],"Slot token/schema differs");
                int index=Convert.ToInt32(state["index"]),count=Convert.ToInt32(request["slot_count"]);
                string phase=(string)state["phase"];
                long launch=Convert.ToInt64(state["launch_monotonic_ns"]),deadline=Convert.ToInt64(state["hard_deadline_ns"]);
                double timeout=Convert.ToDouble(request["slot_timeout_seconds"]);
                Need(timeout>0 && timeout<=45 && index>=0 && index<count && launch>0 && deadline>launch &&
                     deadline-launch<=timeout*1000000000d && launch<=Nanoseconds() &&
                     (phase=="active" || phase=="terminal"),"Slot bounds/phase differ");
                if(index==lastIndex)Need(launch==lastLaunch && deadline==lastDeadline && !(lastPhase=="terminal" && phase=="active"),"Slot deadline or phase moved backwards");
                else Need(index==lastIndex+1,"Slot order skipped or regressed");
                if(phase=="terminal") {
                    var child=(Dictionary<string,object>)state["child_identity"];
                    var terminal=(Dictionary<string,object>)state["terminal_identity"];
                    Need(Convert.ToInt64(child["pid"])==Convert.ToInt64(terminal["pid"]) &&
                         Convert.ToInt64(child["birth_filetime_100ns"])==Convert.ToInt64(terminal["birth_filetime_100ns"]) &&
                         Convert.ToInt64(terminal["exit_filetime_100ns"])>Convert.ToInt64(terminal["birth_filetime_100ns"]),"Slot terminal identity differs");
                }
                latest=state; lastIndex=index; lastLaunch=launch; lastDeadline=deadline; lastPhase=phase;
                string bytes=Json.Serialize(state);
                if(bytes!=lastBytes) {
                    Write(Path.Combine(directory,"slot_observation_"+index.ToString("D3")+"_"+phase+".json"),D("state",state,"observed_qpc_ns",Nanoseconds()));
                    lastBytes=bytes;
                }
                return phase=="active" && Nanoseconds()>=deadline ? "active_slot_deadline_exceeded" : null;
            } catch(Exception error) {
                Write(Path.Combine(directory,"slot_guard_failure.json"),D("status","failed","error",error.GetType().Name+": "+error.Message,"qpc_ns",Nanoseconds()));
                return "active_slot_state_invalid";
            }
        }
    }

    static int Observe(string requestPath) {
        string requestHash=Hash(requestPath); var request=Read(requestPath); string directory=Path.GetFullPath((string)request["directory"]);
        Directory.CreateDirectory(directory);
        ulong observerMask=Pin(request,"observer_cpu",10);
        double deadline=Convert.ToDouble(request["deadline_seconds"]),grace=Convert.ToDouble(request["confirmation_seconds"]);
        Need(deadline>0 && deadline<=900 && grace>0 && grace<=10,"Bounds rejected");
        long start=Stamp();
        Write(Path.Combine(directory,"watchdog_open.json"),D("status","observer_started","qpc_ticks",start,"qpc_frequency",Stopwatch.Frequency,
            "deadline_seconds",deadline,"confirmation_seconds",grace,"termination_apis_in_observer",false,
            "request_path",Path.GetFullPath(requestPath),"request_sha256",requestHash,"observer_affinity_mask",observerMask,
            "no_next_window_after_unconfirmed_termination",true));
        PROCESS_INFORMATION owner=new PROCESS_INFORMATION(); Dictionary<string,object> ownerIdentity=null; string reason=null;
        var slotGuard=new SlotGuard(request,directory);
        try {
            CheckInputs(request); Need(Hash(requestPath)==requestHash,"Request changed before launch");
            Write(Path.Combine(directory,"watchdog_input_guard.json"),D("status","passed","request_sha256",requestHash,"inputs_sha256",request["inputs_sha256"]));
            // Native bInheritHandles=false prevents inheritance of *any* caller pipes,
            // not merely replacement of the three standard streams.
            var startup=new STARTUPINFOEX(); startup.startup.cb=Marshal.SizeOf(typeof(STARTUPINFO));
            Need(CreateProcess(Self(),new StringBuilder(Quote(Self())+" --owner "+Quote(Path.GetFullPath(requestPath))+" "+requestHash),
                IntPtr.Zero,IntPtr.Zero,false,CREATE_NO_WINDOW,IntPtr.Zero,Path.GetDirectoryName(Self()),ref startup,out owner),"Create owner without inherited handles");
            ownerIdentity=Identity(owner.process,owner.pid);
            Write(Path.Combine(directory,"owner_identity.json"),ownerIdentity);
            while(Seconds(start)<deadline) {
                reason=slotGuard.Poll(); if(reason!=null)break;
                if(WaitForSingleObject(owner.process,10)==WAIT_OBJECT_0) { reason="owner_exited"; break; }
                if(File.Exists(Path.Combine(directory,"fixture_force_deadline.flag"))) { reason="fixture_forced_deadline"; break; }
            }
            if(reason==null)reason="observer_deadline_exceeded";
            // Persist intent before asking a separate process to call any termination API.
            Write(Path.Combine(directory,"watchdog_deadline.json"),D("reason",reason,"qpc_ticks",Stamp(),"wall_seconds",Seconds(start),
                "owner_identity",ownerIdentity,"coordinator_identity",TryRead(Path.Combine(directory,"owner_ready.json")),
                "active_slot_state",slotGuard.latest,"termination_confirmed",false));
            if(WaitForSingleObject(owner.process,0)!=WAIT_OBJECT_0)Write(Path.Combine(directory,"stop_requested.json"),D("reason",reason,"qpc_ticks",Stamp()));
            long confirmation=Stamp();
            while(Seconds(confirmation)<grace && WaitForSingleObject(owner.process,10)!=WAIT_OBJECT_0) { }
            var terminal=TryRead(Path.Combine(directory,"owner_terminal.json"));
            bool ownerExited=WaitForSingleObject(owner.process,0)==WAIT_OBJECT_0;
            bool confirmed=ownerExited && terminal!=null && Convert.ToBoolean(terminal["job_empty_confirmed"]);
            var result=D("status",confirmed ? "owned_tree_terminal" : "termination_pending","reason",reason,
                "wall_seconds",Seconds(start),"observer_deadline_seconds",deadline,"confirmation_seconds",grace,
                "owner_identity",ownerIdentity,"owner_terminal",terminal,"termination_confirmed",confirmed,
                "owner_exited",ownerExited,"next_window_allowed",false,"gpu_contexts_created_by_watchdog",0,
                "request_sha256",requestHash,"input_guard_sha256",Hash(Path.Combine(directory,"watchdog_input_guard.json")));
            bool closingGuard=true;
            try { CheckInputs(request); Need(Hash(requestPath)==requestHash,"Request changed during execution"); }
            catch(Exception error) { closingGuard=false; result.Add("closing_guard_error",error.GetType().Name+": "+error.Message); }
            result.Add("closing_input_guard_passed",closingGuard);
            if(!closingGuard)result["status"]="verification_failed";
            if(ownerExited) { uint code; Need(GetExitCodeProcess(owner.process,out code),"Owner exit code"); result.Add("owner_exit_code",code); }
            Write(Path.Combine(directory,"watchdog_terminal.json"),result);
            return !closingGuard ? 4 : confirmed ? 0 : 2;
        } catch(Exception error) {
            Write(Path.Combine(directory,"watchdog_terminal.json"),D("status","termination_pending","reason","observer_exception",
                "error",error.GetType().Name+": "+error.Message,"owner_identity",ownerIdentity,"wall_seconds",Seconds(start),
                "termination_confirmed",false,"next_window_allowed",false));
            return 3;
        }
        // Observer owns no Job handle and intentionally invokes no CloseHandle/Terminate* cleanup.
    }
    static Dictionary<string,object> TryRead(string path) {
        try { return File.Exists(path) ? Read(path) : null; } catch(IOException) { return null; }
    }
    static int Owner(string requestPath,string requestHash) {
        Need(Hash(requestPath)==requestHash,"Owner request hash differs");
        var request=Read(requestPath); string directory=Path.GetFullPath((string)request["directory"]);
        ulong ownerMask=Pin(request,"owner_cpu",12);
        IntPtr job=IntPtr.Zero,attributes=IntPtr.Zero,jobPointer=IntPtr.Zero; PROCESS_INFORMATION child=new PROCESS_INFORMATION(); bool assigned=false,resumed=false,empty=false;
        string reason="owner_setup_exception"; Dictionary<string,object> born=null; uint exitCode=259; long start=Stamp();
        try {
            CheckInputs(request);
            job=CreateJobObject(IntPtr.Zero,null); Need(job!=IntPtr.Zero,"CreateJobObject");
            var limits=new EXTENDED_LIMIT(); limits.basic.flags=JOB_KILL_ON_CLOSE;
            Need(SetInformationJobObject(job,9,ref limits,(uint)Marshal.SizeOf(typeof(EXTENDED_LIMIT))),"SetInformationJobObject");
            // Windows 10+ JOB_LIST performs assignment atomically with process creation.
            // A failed assignment cannot leave an unassigned suspended coordinator.
            IntPtr size=IntPtr.Zero; InitializeProcThreadAttributeList(IntPtr.Zero,1,0,ref size);
            attributes=Marshal.AllocHGlobal(size); Need(InitializeProcThreadAttributeList(attributes,1,0,ref size),"InitializeProcThreadAttributeList");
            jobPointer=Marshal.AllocHGlobal(IntPtr.Size); Marshal.WriteIntPtr(jobPointer,job);
            Need(UpdateProcThreadAttribute(attributes,0,new IntPtr(0x0002000D),jobPointer,new IntPtr(IntPtr.Size),IntPtr.Zero,IntPtr.Zero),"UpdateProcThreadAttribute JOB_LIST");
            var startup=new STARTUPINFOEX(); startup.startup.cb=Marshal.SizeOf(typeof(STARTUPINFOEX)); startup.attributes=attributes;
            Need(CreateProcess((string)request["application"],new StringBuilder((string)request["command_line"]),IntPtr.Zero,IntPtr.Zero,false,
                CREATE_SUSPENDED|CREATE_NO_WINDOW|0x00080000,IntPtr.Zero,(string)request["working_directory"],ref startup,out child),"CreateProcess atomically in Job and suspended");
            born=Identity(child.process,child.pid);
            Write(Path.Combine(directory,"owner_created_suspended.json"),D("coordinator_identity",born,"qpc_ticks",Stamp(),"resumed",false));
            Need(IsProcessInJob(child.process,job,out assigned) && assigned,"IsProcessInJob");
            Write(Path.Combine(directory,"owner_ready.json"),D("coordinator_identity",born,"qpc_ticks",Stamp(),"assigned_before_resume",true,
                "job_kill_on_close",true,"job_handle_inherited",false,"termination_api_process","job_owner_only","owner_affinity_mask",ownerMask));
            Need(ResumeThread(child.thread)!=0xffffffff,"ResumeThread"); resumed=true;
            if(request.ContainsKey("fixture_owner_abort_after_resume") && Convert.ToBoolean(request["fixture_owner_abort_after_resume"])) {
                Thread.Sleep(500); Environment.Exit(23);
            }
            while(true) {
                uint wait=WaitForSingleObject(child.process,10); Need(wait==WAIT_OBJECT_0 || wait==WAIT_TIMEOUT,"Child wait");
                if(wait==WAIT_OBJECT_0) { reason="coordinator_exited"; break; }
                if(File.Exists(Path.Combine(directory,"stop_requested.json"))) { reason="observer_stop_requested"; break; }
                if(Seconds(start)>Convert.ToDouble(request["deadline_seconds"])+Convert.ToDouble(request["confirmation_seconds"])+5) {
                    reason="owner_failsafe_deadline"; break;
                }
            }
            if(Active(job)!=0) {
                Write(Path.Combine(directory,"owner_termination_intent.json"),D("reason",reason,"qpc_ticks",Stamp(),"coordinator_identity",born,
                    "active_job_processes",Active(job),"termination_confirmed",false));
                // CPU fixture simulates a stuck native termination without blocking the observer.
                int delay=request.ContainsKey("fixture_termination_delay_ms") ? Convert.ToInt32(request["fixture_termination_delay_ms"]) : 0;
                if(delay>0)Thread.Sleep(delay);
                bool ok=TerminateJobObject(job,TERMINATION_CODE);
                Write(Path.Combine(directory,"owner_termination_return.json"),D("ok",ok,"qpc_ticks",Stamp(),"win32_error",ok ? 0 : Marshal.GetLastWin32Error()));
                long until=Stamp(); while(Active(job)!=0 && Seconds(until)<4)Thread.Sleep(10);
            }
            empty=Active(job)==0;
            if(empty && child.process!=IntPtr.Zero)GetExitCodeProcess(child.process,out exitCode);
            Write(Path.Combine(directory,"owner_terminal.json"),D("status",empty ? "job_empty" : "termination_pending","reason",reason,
                "job_empty_confirmed",empty,"coordinator_terminal",Identity(child.process,child.pid),"coordinator_exit_code",exitCode,
                "assigned_before_resume",assigned && resumed,"qpc_ticks",Stamp()));
            return empty ? 0 : 2;
        } catch(Exception error) {
            Write(Path.Combine(directory,"owner_exception.json"),D("status","termination_pending","error",error.GetType().Name+": "+error.Message,
                "assigned",assigned,"resumed",resumed,"coordinator_identity",born,"qpc_ticks",Stamp()));
            return 3;
        } finally {
            // These APIs may hang: exclusively in owner, after persistent evidence.
            if(job!=IntPtr.Zero)CloseHandle(job);
            if(child.thread!=IntPtr.Zero)CloseHandle(child.thread);
            if(child.process!=IntPtr.Zero)CloseHandle(child.process);
            if(attributes!=IntPtr.Zero) { DeleteProcThreadAttributeList(attributes); Marshal.FreeHGlobal(attributes); }
            if(jobPointer!=IntPtr.Zero)Marshal.FreeHGlobal(jobPointer);
        }
    }
    static int Synthetic(string mode,string directory) {
        if(mode=="leaf") { Thread.Sleep(60000); return 0; }
        if(mode=="grandchild" || mode=="fail_with_child") {
            var setup=new ProcessStartInfo(Self(),"--synthetic leaf "+Quote(directory)); setup.UseShellExecute=false; setup.CreateNoWindow=true;
            var leaf=Process.Start(setup);
            Write(Path.Combine(directory,"synthetic_leaf_identity.json"),Identity(leaf.Handle,(uint)leaf.Id));
            if(mode=="fail_with_child")return 19;
        }
        if(mode=="success") { Thread.Sleep(100); return 0; }
        if(mode=="failure")return 17;
        if(mode=="slot_deadline" || mode=="slot_extend" || mode=="slot_bad_identity") {
            long start=Nanoseconds();
            var identity=Identity(GetCurrentProcess(),(uint)Process.GetCurrentProcess().Id);
            if(mode=="slot_bad_identity")identity["birth_filetime_100ns"]=1L;
            var state=D("schema","rfm-v8-outer-slot-1","run_token","cpu-fixture-token","index",0,"phase","active",
                "launch_monotonic_ns",start,"hard_deadline_ns",start+300000000L,"coordinator",identity);
            string path=Path.Combine(directory,"slot_state.json"); Write(path,state);
            if(mode=="slot_extend") {
                Thread.Sleep(100); state["hard_deadline_ns"]=start+400000000L;
                Write(path+".next",state); File.Replace(path+".next",path,null);
            }
        }
        Thread.Sleep(60000); return 0;
    }
    public static int Main(string[] args) {
        try {
            if(args.Length==2 && args[0]=="--observe")return Observe(args[1]);
            if(args.Length==3 && args[0]=="--owner")return Owner(args[1],args[2]);
            if(args.Length==3 && args[0]=="--synthetic")return Synthetic(args[1],args[2]);
            throw new ArgumentException("Expected --observe request, --owner request, or --synthetic mode directory");
        } catch(Exception error) { Console.Error.WriteLine(error); return 99; }
    }
}
