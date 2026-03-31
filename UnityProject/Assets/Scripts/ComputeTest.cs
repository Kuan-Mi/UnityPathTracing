using UnityEngine;
using System.Runtime.InteropServices;

public class ComputeTest : MonoBehaviour
{
    public ComputeShader shader;
    
    // C# 端的结构体定义，必须使用 Sequential 布局
    [StructLayout(LayoutKind.Sequential)]
    public struct MyStruct
    {
        public Vector4 data;
    }

    void Start()
    {
        if (shader == null) return;

        // 1. 准备数据
        MyStruct myData = new MyStruct();
        myData.data = new Vector4(0.75f, 0, 0, 0); // 我们期望纹理颜色变成 0.75 (灰色)

        // 2. 创建 Constant Buffer
        // 常量缓冲区的大小必须是 16 的倍数
        int stride = Marshal.SizeOf(typeof(MyStruct));
        int bufferSize = (stride + 15) / 16 * 16;
        ComputeBuffer cb = new ComputeBuffer(1, bufferSize, ComputeBufferType.Constant);
        cb.SetData(new MyStruct[] { myData });

        // 3. 创建测试纹理 (RFloat 格式，对应 Shader 中的 float)
        RenderTexture rt = new RenderTexture(256, 256, 0, RenderTextureFormat.RFloat);
        rt.enableRandomWrite = true;
        rt.Create();

        // 4. 设置参数并运行
        int kernel = shader.FindKernel("CSMain");
        
        // 注意：这里名字必须是 "test"，对应 Shader 里的 ConstantBuffer<my_struct> test
        shader.SetConstantBuffer("test", cb, 0, bufferSize);
        shader.SetTexture(kernel, "_Target", rt);
        
        shader.Dispatch(kernel, 256 / 8, 256 / 8, 1);

        // 5. 验证结果 (读取纹理的一个像素)
        VerifyPixel(rt);

        // 6. 释放资源
        cb.Release();
        // rt.Release(); // 如果要在 Game 视图看结果，先别 Release
        
        // 将结果显示在场景物体上供观察（可选）
        GetComponent<Renderer>().material.mainTexture = rt;
    }

    void VerifyPixel(RenderTexture rt)
    {
        Texture2D temp = new Texture2D(rt.width, rt.height, TextureFormat.RFloat, false);
        RenderTexture.active = rt;
        temp.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        temp.Apply();
        
        float pixelValue = temp.GetPixel(0, 0).r;
        Debug.Log($"<color=green>Shader 运行成功！像素 (0,0) 的值是: {pixelValue}</color> (期望值: 0.75)");
        
        RenderTexture.active = null;
        Destroy(temp);
    }
}