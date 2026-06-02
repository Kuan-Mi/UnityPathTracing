using System.Text;

namespace NativeRender
{
    /// <summary>
    /// Helpers for reading the DXIL container header that DXC emits. A DXIL blob reuses the legacy
    /// DXBC container layout: a 4-byte <c>'DXBC'</c> fourcc immediately followed by a 16-byte hash
    /// digest (<c>DxilContainerHash</c>). DXC writes this digest — the same value PIX / RenderDoc
    /// display as the "Shader Hash" — when the validator runs; with <c>-Zsb</c> it is derived from
    /// the output binary, with <c>-Zss</c> from the source.
    /// </summary>
    internal static class DxilContainerUtil
    {
        // 'DXBC' stored little-endian: 'D'=0x44, 'X'=0x58, 'B'=0x42, 'C'=0x43.
        private const uint ContainerFourCC = 0x43425844;

        // Container layout: [0..4) fourcc, [4..20) 16-byte hash digest, then version / size / parts.
        private const int FourCCSize   = 4;
        private const int HashByteCount = 16;

        /// <summary>
        /// Returns the 16-byte DXIL container hash as a 32-char lowercase hex string, or "" if the
        /// bytes are not a valid DXIL container (too short or missing the 'DXBC' fourcc).
        /// </summary>
        public static string ExtractHashHex(byte[] dxil)
        {
            if (dxil == null || dxil.Length < FourCCSize + HashByteCount) return "";

            uint fourcc = (uint)(dxil[0] | (dxil[1] << 8) | (dxil[2] << 16) | (dxil[3] << 24));
            if (fourcc != ContainerFourCC) return "";

            var sb = new StringBuilder(HashByteCount * 2);
            for (int i = FourCCSize; i < FourCCSize + HashByteCount; i++)
                sb.Append(dxil[i].ToString("x2"));
            return sb.ToString();
        }
    }
}
