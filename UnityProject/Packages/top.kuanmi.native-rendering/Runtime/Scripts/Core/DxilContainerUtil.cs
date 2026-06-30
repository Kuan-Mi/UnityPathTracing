using System.Text;

namespace NativeRender
{
    /// <summary>
    /// Helpers for reading the DXIL container that DXC emits. A DXIL blob reuses the legacy DXBC
    /// container layout: a 4-byte <c>'DXBC'</c> fourcc, a 16-byte container-integrity checksum, a
    /// version/size header, then a table of named parts. The value PIX / RenderDoc display as the
    /// "Shader Hash" is the dedicated <c>HASH</c> part (<c>DxilShaderHash</c>) — NOT the 16 bytes in
    /// the header, which are a checksum over the whole container (including any embedded debug) and
    /// differ from the reported shader hash. With <c>-Zsb</c> the shader hash is derived from the
    /// output binary, with <c>-Zss</c> from the source.
    /// </summary>
    internal static class DxilContainerUtil
    {
        // 'DXBC' stored little-endian: 'D'=0x44, 'X'=0x58, 'B'=0x42, 'C'=0x43.
        private const uint ContainerFourCC = 0x43425844;

        // 'HASH' part fourcc (little-endian): 'H'=0x48, 'A'=0x41, 'S'=0x53, 'H'=0x48.
        private const uint HashPartFourCC = 0x48534148;

        // Container header: [0..4) fourcc, [4..20) integrity checksum, [20..24) version,
        // [24..28) container size, [28..32) part count, [32..) uint32 part offsets.
        private const int PartCountOffset = 28;
        private const int PartTableOffset = 32;

        // Each part: [0..4) fourcc, [4..8) data size, [8..) data. The HASH part's data is a
        // DxilShaderHash { uint32 Flags; byte Digest[16] }, so the digest starts 4 bytes in.
        private const int PartHeaderSize         = 8;
        private const int HashDigestOffsetInPart = 4;
        private const int HashByteCount          = 16;

        /// <summary>
        /// Returns the DXIL shader hash (the <c>HASH</c> part's 16-byte digest) as a 32-char
        /// lowercase hex string — the same value PIX / RenderDoc and DXC report for this shader.
        /// Returns "" if the bytes are not a valid DXIL container or carry no <c>HASH</c> part
        /// (e.g. an unsigned blob, or one compiled with validation disabled).
        /// </summary>
        public static string ExtractHashHex(byte[] dxil)
        {
            if (dxil == null || dxil.Length < PartTableOffset) return "";
            if (ReadU32(dxil, 0) != ContainerFourCC) return "";

            uint partCount = ReadU32(dxil, PartCountOffset);
            if (PartTableOffset + (long)partCount * 4 > dxil.Length) return "";

            for (uint i = 0; i < partCount; i++)
            {
                int partOffset = (int)ReadU32(dxil, PartTableOffset + (int)i * 4);
                if (partOffset < 0 || partOffset + PartHeaderSize > dxil.Length) continue;
                if (ReadU32(dxil, partOffset) != HashPartFourCC) continue;

                int digest = partOffset + PartHeaderSize + HashDigestOffsetInPart;
                if (digest + HashByteCount > dxil.Length) return "";

                var sb = new StringBuilder(HashByteCount * 2);
                for (int b = 0; b < HashByteCount; b++)
                    sb.Append(dxil[digest + b].ToString("x2"));
                return sb.ToString();
            }

            return "";
        }

        private static uint ReadU32(byte[] d, int o) =>
            (uint)(d[o] | (d[o + 1] << 8) | (d[o + 2] << 16) | (d[o + 3] << 24));
    }
}