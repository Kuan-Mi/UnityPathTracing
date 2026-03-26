using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    public class LightCollector : IDisposable
    {
        private GraphicsBuffer _spotLightBuffer;
        private GraphicsBuffer _areaLightBuffer;
        private GraphicsBuffer _pointLightBuffer;

        private readonly List<SpotLightData> _spotLightList = new();
        private readonly List<AreaLightData> _areaLightList = new();
        private readonly List<PointLightData> _pointLightList = new();

        public GraphicsBuffer SpotLightBuffer => _spotLightBuffer;
        public GraphicsBuffer AreaLightBuffer => _areaLightBuffer;
        public GraphicsBuffer PointLightBuffer => _pointLightBuffer;

        public int SpotCount { get; private set; }
        public int AreaCount { get; private set; }
        public int PointCount { get; private set; }

        public void Collect()
        {
            CollectSpotLights();
            CollectAreaLights();
            CollectPointLights();
        }

        private void CollectSpotLights()
        {
            _spotLightList.Clear();

            var allLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            allLights = Array.Empty<Light>();
            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Spot) continue;

                Vector3 pos = light.transform.position;
                Vector3 dir = light.transform.forward.normalized;
                Color fc = light.color * light.intensity;

                float outerHalf = light.spotAngle * 0.5f * Mathf.Deg2Rad;
                float innerHalf = light.innerSpotAngle * 0.5f * Mathf.Deg2Rad;

                _spotLightList.Add(new SpotLightData
                {
                    position = pos,
                    range = light.range,
                    direction = dir,
                    cosOuterAngle = Mathf.Cos(outerHalf),
                    color = new Vector3(fc.r, fc.g, fc.b),
                    cosInnerAngle = Mathf.Cos(innerHalf),
                });
            }

            SpotCount = _spotLightList.Count;
            int bufferCount = Mathf.Max(SpotCount, 1);
            if (_spotLightBuffer == null || _spotLightBuffer.count < bufferCount)
            {
                _spotLightBuffer?.Release();
                _spotLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, bufferCount,
                    Marshal.SizeOf<SpotLightData>());
            }

            if (SpotCount > 0)
                _spotLightBuffer.SetData(_spotLightList);
        }

        private void CollectAreaLights()
        {
            _areaLightList.Clear();

            var allLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            allLights = Array.Empty<Light>();
            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Rectangle && light.type != LightType.Disc) continue;

                Color fc = light.color * light.intensity;
                Vector2 sz = light.areaSize;
                bool isDisc = light.type == LightType.Disc;

                _areaLightList.Add(new AreaLightData
                {
                    position = light.transform.position,
                    halfWidth = isDisc ? sz.x : sz.x * 0.5f,
                    right = light.transform.right.normalized,
                    halfHeight = isDisc ? 0f : sz.y * 0.5f,
                    up = light.transform.up.normalized,
                    lightType = isDisc ? 1f : 0f,
                    color = new Vector3(fc.r, fc.g, fc.b),
                    pad2 = 0f,
                });
            }

            AreaCount = _areaLightList.Count;
            int bufferCount = Mathf.Max(AreaCount, 1);
            if (_areaLightBuffer == null || _areaLightBuffer.count < bufferCount)
            {
                _areaLightBuffer?.Release();
                _areaLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, bufferCount,
                    Marshal.SizeOf<AreaLightData>());
            }

            if (AreaCount > 0)
                _areaLightBuffer.SetData(_areaLightList);
        }

        private void CollectPointLights()
        {
            _pointLightList.Clear();

            var allLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            allLights = Array.Empty<Light>();
            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Point) continue;

                Color fc = light.color * light.intensity;

                var plr = light.GetComponent<PointLightRadius>();
                float radius = plr != null ? Mathf.Max(0f, plr.radius) : 0f;

                _pointLightList.Add(new PointLightData
                {
                    position = light.transform.position,
                    range = light.range,
                    color = new Vector3(fc.r, fc.g, fc.b),
                    radius = radius,
                });
            }

            PointCount = _pointLightList.Count;
            int bufferCount = Mathf.Max(PointCount, 1);
            if (_pointLightBuffer == null || _pointLightBuffer.count < bufferCount)
            {
                _pointLightBuffer?.Release();
                _pointLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, bufferCount,
                    Marshal.SizeOf<PointLightData>());
            }

            if (PointCount > 0)
                _pointLightBuffer.SetData(_pointLightList);
        }

        public void Dispose()
        {
            _spotLightBuffer?.Release();
            _spotLightBuffer = null;

            _areaLightBuffer?.Release();
            _areaLightBuffer = null;

            _pointLightBuffer?.Release();
            _pointLightBuffer = null;
        }
    }
}
