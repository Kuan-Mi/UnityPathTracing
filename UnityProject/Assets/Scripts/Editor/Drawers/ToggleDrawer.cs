using UnityEngine;
using UnityEditor;

[CustomPropertyDrawer(typeof(ToggleAttribute))]
public class ToggleDrawer : PropertyDrawer {
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label) {
        if (property.propertyType == SerializedPropertyType.Integer) {
            bool value = property.intValue != 0;
            value             = EditorGUI.Toggle(position, label, value);
            property.intValue = value ? 1 : 0;
        } else {
            EditorGUI.LabelField(position, label.text, "Use Toggle with int or uint.");
        }
    }
}