using System.Runtime.InteropServices;

public enum CheckerboardMode : int
{
    Off = 0,
    Black = 1, // 根据 C++ 实际定义补充
    White = 2 // 根据 C++ 实际定义补充
}

[StructLayout(LayoutKind.Sequential)]
public struct ReSTIRDIStaticParameters
{
    public uint NeighborOffsetCount; // uint32_t -> uint
    public uint RenderWidth; // uint32_t -> uint
    public uint RenderHeight; // uint32_t -> uint

    public CheckerboardMode CheckerboardSamplingMode; // Enum 通常对应 int
}