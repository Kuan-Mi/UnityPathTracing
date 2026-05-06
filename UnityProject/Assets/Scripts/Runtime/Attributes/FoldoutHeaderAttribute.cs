using UnityEngine;
using System;

// 用来代替原生 [Header]
[AttributeUsage(AttributeTargets.Field)]
public class FoldoutHeaderAttribute : PropertyAttribute
{
    public string Name;
    public FoldoutHeaderAttribute(string name)
    {
        this.Name = name;
    }
}