using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

/*
* Copyright 2021 By Colin Leet (https://leet.games/)

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain.We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors.We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to<http://unlicense.org/>
*/

namespace LeetProfiling {
    /// <summary>
    /// This is a testing script for find out which of Unity's samplers can report GPU times on your given platform.
    ///
    /// This script creates recorders for all of available <see cref="Sampler.GetNames(List{string})"/>.
    /// It then runs for a bit and checks which recorders report positive values for <see cref="Recorder.gpuElapsedNanoseconds"/>.
    ///
    /// Note: Only the editor and development builds can use Recorders as of Unity 2020.2
    /// </summary>
    public class TestGPURecorders : MonoBehaviour {

        [Tooltip("Enables the recorder GPU testing.")]
        public bool RunTestInStart = true;

        [Tooltip("Seconds which will be waited before test before initiating the profilers and allNames lists.")]
        [Range(3, 9f)]
        public float WaitTimeBeforeInitiation = 3f;

        [Tooltip("Seconds which will be waited before running the test for GPU profilers.")]
        [Range(1, 30f)]
        public float WaitTimeBeforeTest = 5f;

        [Tooltip("Runs the test twice...")]
        public bool RunTestTwice = true;

        [Header("Runtime Test Results")]
        [Tooltip("Number of recorders which was retrieved with Sampler.GetNames")]
        public int NumRecorders = 0;

        [Tooltip("Number of recorders which reported being valid.")]
        public int ValidRecorders = 0;

        [Tooltip("Number of recorders which have positive GPU times.")]
        public int ActiveGPURecorders = 0;

        /// <summary>
        /// List of names generated from <see cref="Sampler.GetNames(List{string})"/>.
        /// </summary>
        private List<string> allNames = new List<string>();

        /// <summary>
        /// List of all recorders generated from <see cref="Sampler.GetNames(List{string})"/>.
        /// </summary>
        private List<Recorder> allRecorders = new List<Recorder>();

        /// <summary>
        /// List of the keys for <see cref="GpuRecorders"/>.
        /// </summary>
        public List<string> GpuNamesActive = new List<string>();

        /// <summary>
        /// This is a dictionary of all of the actively recording GPU recorders.
        /// </summary>
        public Dictionary<string, Recorder> GpuRecorders = new Dictionary<string, Recorder>();

        private void Start() {
            if ( RunTestInStart ) {
                StartCoroutine(StartTestsWithDelays());
            }
        }

        /// <summary>
        /// Initiates all of the supported recorders.
        /// </summary>
        public void InitiateAllRecorders() {
            // Only run in editor or debug builds.
            if ( !( Application.isEditor || Debug.isDebugBuild ) ) return;

            allNames.Clear();
            allRecorders.Clear();
            GpuNamesActive.Clear();
            GpuRecorders.Clear();

            Sampler.GetNames(allNames);
            NumRecorders = allNames.Count;
            ValidRecorders = 0;
            ActiveGPURecorders = 0;

            int lenRecorders = allNames.Count;
            for ( int i = 0; i < lenRecorders; i++ ) {
                allRecorders.Add(Recorder.Get(allNames[i]));
                allRecorders[i].enabled = true;
            }
        }

        /// <summary>
        /// Waits a period then runs the test, then logs the results.
        /// </summary>
        public IEnumerator StartTestsWithDelays() {
            // Only run in editor or debug builds.
            if ( !( Application.isEditor || Debug.isDebugBuild ) ) yield break;

            // Wait for the scene to get going.
            yield return new WaitForSecondsRealtime(WaitTimeBeforeInitiation);

            // Init the recorder vars.
            InitiateAllRecorders();

            // Wait for the scene to get going.
            for ( int i = 0; i < 2; i++ ) {

                yield return new WaitForSecondsRealtime(WaitTimeBeforeTest);

                AddNewGPURecordingProfilers();

                yield return null;

                LogResults();

                if ( !RunTestTwice ) yield break;
            }
        }


        /// <summary>
        /// Tests which GPU times have positive values.
        /// This may be called multiple times outside of <see cref="StartTestsWithDelays"/>.
        /// </summary>
        public void AddNewGPURecordingProfilers() {
            // Determine all of the active gpu recorders
            int lenRecorders = allNames.Count;
            for ( int i = 0; i < lenRecorders; i++ ) {
                if ( allRecorders[i].isValid ) {

                    ValidRecorders++;

                    if ( allRecorders[i].gpuElapsedNanoseconds > 0 ) {
                        if ( !GpuRecorders.ContainsKey(allNames[i]) ) {

                            GpuNamesActive.Add(allNames[i]);
                            GpuRecorders[allNames[i]] = allRecorders[i];

                            ActiveGPURecorders++;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Lists all of the active gpu recorders.
        /// </summary>
        public void LogResults() {
            Debug.LogFormat("{0} Recorders are reporting positive times for the GPU... Listing <{0}>: ", ActiveGPURecorders);
            int countActive = GpuNamesActive.Count;
            for ( int i = 0; i < countActive; i++ ) {
                Debug.LogFormat("{0}: {1:N5} ms",
                    GpuNamesActive[i],
                    GpuRecorders[GpuNamesActive[i]].gpuElapsedNanoseconds * 1e-6);
            }
        }
    }
}