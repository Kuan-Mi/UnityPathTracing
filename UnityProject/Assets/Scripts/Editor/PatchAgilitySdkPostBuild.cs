using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

public class PatchAgilitySdkPostBuild : IPostprocessBuildWithReport
{
    public int callbackOrder => 0;

    public void OnPostprocessBuild(BuildReport report)
    {
        string exePath = report.summary.outputPath;
        if (!exePath.EndsWith(".exe"))
        {
            Debug.Log("[AgilitySDK Patch] Not a Windows build, skipping.");
            return;
        }

        Debug.Log($"[AgilitySDK Patch] Patching {exePath} ...");

        byte[] bytes = System.IO.File.ReadAllBytes(exePath);
        byte[] pattern = { 0x6A, 0x02, 0x00, 0x00 };
        byte[] replace = { 0x6B, 0x02, 0x00, 0x00 };
        int count = 0;

        for (int i = 0; i <= bytes.Length - pattern.Length; i++)
        {
            if (bytes[i] == pattern[0] && bytes[i + 1] == pattern[1] &&
                bytes[i + 2] == pattern[2] && bytes[i + 3] == pattern[3])
            {
                Debug.Log($"[AgilitySDK Patch] Found match at offset 0x{i:X8}");
                for (int j = 0; j < replace.Length; j++)
                    bytes[i + j] = replace[j];
                count++;
            }
        }

        if (count == 0)
        {
            Debug.LogWarning("[AgilitySDK Patch] Pattern 6A 02 00 00 not found in exe.");
        }
        else
        {
            System.IO.File.WriteAllBytes(exePath, bytes);
            Debug.Log($"[AgilitySDK Patch] Patched {count} occurrence(s). (SDK 618 -> 619)");
        }
    }
}
