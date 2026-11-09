// Query-only Windows process handles for exact FILETIME sampling.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

public static class PreparedProcessSnapshot
{
    private const uint ProcessSnapshot = 2;
    private const uint QueryLimitedInformation = 0x1000;
    private const int NoMoreFiles = 18;

    [StructLayout(LayoutKind.Sequential)]
    private struct FileTime
    {
        public uint Low, High;
        public long Ticks { get { return ((long)High << 32) | Low; } }
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct ProcessEntry
    {
        public uint Size, Usage, Pid;
        public IntPtr DefaultHeap;
        public uint ModuleId, Threads, ParentPid;
        public int Priority;
        public uint Flags;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)] public string Image;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr CreateToolhelp32Snapshot(uint flags, uint pid);
    [DllImport("kernel32.dll", EntryPoint = "Process32FirstW", SetLastError = true, CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern bool Process32First(IntPtr snapshot, ref ProcessEntry entry);
    [DllImport("kernel32.dll", EntryPoint = "Process32NextW", SetLastError = true, CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern bool Process32Next(IntPtr snapshot, ref ProcessEntry entry);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern int GetProcessId(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessTimes(IntPtr handle, out FileTime birth, out FileTime exit,
                                               out FileTime kernel, out FileTime user);
    [DllImport("kernel32.dll", EntryPoint = "QueryFullProcessImageNameW", SetLastError = true, CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern bool QueryFullProcessImageName(IntPtr handle, uint flags, StringBuilder name, ref uint length);

    public sealed class Row
    {
        public int pid { get; set; }
        public int? enumerated_parent_pid { get; set; }
        public long birth_filetime_100ns { get; set; }
        public long exit_filetime_100ns { get; set; }
        public long kernel_cpu_100ns { get; set; }
        public long user_cpu_100ns { get; set; }
        public string observation { get; set; }
    }

    public sealed class Issue
    {
        public int pid { get; set; }
        public string stage { get; set; }
        public int win32_error { get; set; }
    }

    public sealed class Sample
    {
        public List<Row> python { get; } = new List<Row>();
        public List<Issue> unresolved_python { get; } = new List<Issue>();
    }

    private sealed class Tracked
    {
        public Row Identity;
        public IntPtr Handle;
    }

    private static Dictionary<Tuple<int, long>, Tracked> retained = new Dictionary<Tuple<int, long>, Tracked>();

    private static bool IsPython(string executable)
    {
        return !String.IsNullOrEmpty(executable) && executable.StartsWith("python", StringComparison.OrdinalIgnoreCase);
    }

    private static Row ReadTimes(IntPtr handle, int? parent, string observation)
    {
        int pid = GetProcessId(handle);
        if (pid == 0) throw new Win32Exception(Marshal.GetLastWin32Error());
        FileTime birth, exit, kernel, user;
        if (!GetProcessTimes(handle, out birth, out exit, out kernel, out user))
            throw new Win32Exception(Marshal.GetLastWin32Error());
        return new Row { pid = pid, enumerated_parent_pid = parent, birth_filetime_100ns = birth.Ticks,
            exit_filetime_100ns = exit.Ticks, kernel_cpu_100ns = kernel.Ticks, user_cpu_100ns = user.Ticks,
            observation = observation };
    }

    public static Row IdentityFromHandle(IntPtr handle)
    {
        return ReadTimes(handle, null, "owned_handle");
    }

    private static void ReadCandidate(ProcessEntry entry, Sample sample, Dictionary<Tuple<int, long>, Tracked> next)
    {
        int pid = (int)entry.Pid;
        var handle = OpenProcess(QueryLimitedInformation, false, pid);
        if (handle == IntPtr.Zero)
        {
            sample.unresolved_python.Add(new Issue { pid = pid, stage = "open_enumerated", win32_error = Marshal.GetLastWin32Error() });
            return;
        }
        string stage = "read_enumerated_times";
        try
        {
            Row row = ReadTimes(handle, (int)entry.ParentPid, "enumerated");
            var key = Tuple.Create(row.pid, row.birth_filetime_100ns);
            if (!retained.ContainsKey(key))
            {
                stage = "identify_enumerated_image";
                var image = new StringBuilder(32768);
                uint length = (uint)image.Capacity;
                if (!QueryFullProcessImageName(handle, 0, image, ref length))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                if (!IsPython(Path.GetFileName(image.ToString())))
                {
                    sample.unresolved_python.Add(new Issue { pid = pid, stage = "image_replaced_after_enumeration" });
                    return;
                }
            }
            sample.python.Add(row);
            if (row.exit_filetime_100ns == 0)
            {
                next.Add(key, new Tracked { Identity = row, Handle = handle });
                handle = IntPtr.Zero;
            }
        }
        catch (Win32Exception error)
        {
            sample.unresolved_python.Add(new Issue { pid = pid, stage = stage, win32_error = error.NativeErrorCode });
        }
        finally
        {
            if (handle != IntPtr.Zero) CloseHandle(handle);
        }
    }

    public static Sample Capture()
    {
        var sample = new Sample();
        var next = new Dictionary<Tuple<int, long>, Tracked>();
        var snapshot = CreateToolhelp32Snapshot(ProcessSnapshot, 0);
        if (snapshot == new IntPtr(-1)) throw new Win32Exception(Marshal.GetLastWin32Error());
        try
        {
            var entry = new ProcessEntry { Size = (uint)Marshal.SizeOf(typeof(ProcessEntry)) };
            if (!Process32First(snapshot, ref entry)) throw new Win32Exception(Marshal.GetLastWin32Error());
            do
            {
                if (IsPython(entry.Image)) ReadCandidate(entry, sample, next);
            }
            while (Process32Next(snapshot, ref entry));
            int error = Marshal.GetLastWin32Error();
            if (error != NoMoreFiles) throw new Win32Exception(error);
            var seen = new HashSet<Tuple<int, long>>();
            foreach (var row in sample.python) seen.Add(Tuple.Create(row.pid, row.birth_filetime_100ns));
            foreach (var pair in retained)
            {
                if (seen.Contains(pair.Key)) continue;
                var old = pair.Value;
                try
                {
                    var row = ReadTimes(old.Handle, old.Identity.enumerated_parent_pid, "retained");
                    if (row.exit_filetime_100ns != 0) row.observation = "retained_exit";
                    else next.Add(pair.Key, new Tracked { Identity = row, Handle = old.Handle });
                    sample.python.Add(row);
                }
                catch (Win32Exception ex)
                {
                    sample.unresolved_python.Add(new Issue { pid = old.Identity.pid, stage = "read_retained_times", win32_error = ex.NativeErrorCode });
                }
            }
        }
        catch
        {
            foreach (var pair in next)
                if (!retained.ContainsKey(pair.Key) || retained[pair.Key].Handle != pair.Value.Handle) CloseHandle(pair.Value.Handle);
            throw;
        }
        finally { CloseHandle(snapshot); }
        foreach (var pair in retained)
            if (!next.ContainsKey(pair.Key) || next[pair.Key].Handle != pair.Value.Handle) CloseHandle(pair.Value.Handle);
        retained = next;
        sample.python.Sort((a, b) => a.pid != b.pid ? a.pid.CompareTo(b.pid) : a.birth_filetime_100ns.CompareTo(b.birth_filetime_100ns));
        return sample;
    }

    public static void Close()
    {
        foreach (var item in retained.Values) CloseHandle(item.Handle);
        retained.Clear();
    }
}
