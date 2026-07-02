using System;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class CommandLineBuild
{
    // Invoked via: Unity.exe -batchmode -quit -projectPath <proj>
    //   -executeMethod CommandLineBuild.BuildWindows [-buildPath <exe path>]
    // See build_player.bat at the repo root.
    public static void BuildWindows()
    {
        string[] scenes = EditorBuildSettings.scenes
            .Where(s => s.enabled)
            .Select(s => s.path)
            .ToArray();

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            // relative paths resolve against the project folder
            locationPathName = GetArg("-buildPath") ?? "Build/UnityPathTracing.exe",
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        Debug.Log($"Build result: {report.summary.result}, errors: {report.summary.totalErrors}, " +
                  $"size: {report.summary.totalSize} bytes, output: {report.summary.outputPath}");

        EditorApplication.Exit(report.summary.result == BuildResult.Succeeded ? 0 : 1);
    }

    static string GetArg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == name)
                return args[i + 1];
        return null;
    }
}
