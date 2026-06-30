// #if ENABLE_INPUT_SYSTEM && ENABLE_INPUT_SYSTEM_PACKAGE
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;

// #endif

namespace PathTracing
{
    /// <summary>
    /// Utility Free Camera component.
    /// </summary>
    /// 
    public class MyFreeCamera : MonoBehaviour
    {
        const float k_MouseSensitivityMultiplier = 0.01f;

        /// <summary>
        /// Rotation speed when using a controller.
        /// </summary>
        public float m_LookSpeedController = 120f;

        /// <summary>
        /// Rotation speed when using the mouse.
        /// </summary>
        public float m_LookSpeedMouse = 1f;

        /// <summary>
        /// Movement speed.
        /// </summary>
        public float m_MoveSpeed = 1.0f;

        /// <summary>
        /// Value added to the speed when incrementing.
        /// </summary>
        public float m_MoveSpeedIncrement = 2.5f;

        /// <summary>
        /// Scale factor of the turbo mode.
        /// </summary>
        public float m_Turbo = 10.0f;

        InputActionMap inputMap;
        InputAction mouseLookAction;
        InputAction gamepadLookAction;
        InputAction moveAction;
        InputAction speedAction;
        InputAction yMoveAction;

        void OnEnable()
        {
            RegisterInputs();
        }

        void RegisterInputs()
        {
            inputMap = new InputActionMap("Free Camera");

            mouseLookAction = inputMap.AddAction("mouseLook", binding: "<Mouse>/delta");
            gamepadLookAction = inputMap.AddAction("gamepadLook", binding: "<Gamepad>/rightStick");
            moveAction = inputMap.AddAction("move", binding: "<Gamepad>/leftStick");
            speedAction = inputMap.AddAction("speed", binding: "<Gamepad>/dpad");
            yMoveAction = inputMap.AddAction("yMove");

            moveAction.AddCompositeBinding("Dpad")
                .With("Up", "<Keyboard>/w")
                .With("Up", "<Keyboard>/upArrow")
                .With("Down", "<Keyboard>/s")
                .With("Down", "<Keyboard>/downArrow")
                .With("Left", "<Keyboard>/a")
                .With("Left", "<Keyboard>/leftArrow")
                .With("Right", "<Keyboard>/d")
                .With("Right", "<Keyboard>/rightArrow");
            speedAction.AddCompositeBinding("Dpad")
                .With("Up", "<Keyboard>/home")
                .With("Down", "<Keyboard>/end");
            yMoveAction.AddCompositeBinding("Dpad")
                .With("Up", "<Keyboard>/pageUp")
                .With("Down", "<Keyboard>/pageDown")
                .With("Up", "<Keyboard>/e")
                .With("Down", "<Keyboard>/q")
                .With("Up", "<Gamepad>/rightshoulder")
                .With("Down", "<Gamepad>/leftshoulder");

            inputMap.Enable();
        }

        void OnDisable()
        {
            inputMap?.Disable();
        }

        void OnDestroy()
        {
            inputMap?.Dispose();
            inputMap = null;
            mouseLookAction = null;
            gamepadLookAction = null;
            moveAction = null;
            speedAction = null;
            yMoveAction = null;
        }

        float inputRotateAxisX, inputRotateAxisY;
        float inputChangeSpeed;
        float inputVertical, inputHorizontal, inputYAxis;
        bool leftShiftBoost, leftShift, fire1;

        void UpdateInputs()
        {
            inputRotateAxisX = 0.0f;
            inputRotateAxisY = 0.0f;
            leftShiftBoost = false;
            fire1 = false;

            if (Mouse.current?.rightButton?.isPressed == true)
            {
                leftShiftBoost = true;
                var mouseLookDelta = mouseLookAction.ReadValue<Vector2>();
                inputRotateAxisX = mouseLookDelta.x * m_LookSpeedMouse * k_MouseSensitivityMultiplier;
                inputRotateAxisY = mouseLookDelta.y * m_LookSpeedMouse * k_MouseSensitivityMultiplier;
            }

            var gamepadLookDelta = gamepadLookAction.ReadValue<Vector2>();
            inputRotateAxisX += gamepadLookDelta.x * m_LookSpeedController * k_MouseSensitivityMultiplier;
            inputRotateAxisY += gamepadLookDelta.y * m_LookSpeedController * k_MouseSensitivityMultiplier;

            leftShift = Keyboard.current?.leftShiftKey?.isPressed ?? false;
            fire1 = Mouse.current?.leftButton?.isPressed == true || Gamepad.current?.xButton?.isPressed == true;

            inputChangeSpeed = speedAction.ReadValue<Vector2>().y;

            var moveDelta = moveAction.ReadValue<Vector2>();
            inputVertical = moveDelta.y;
            inputHorizontal = moveDelta.x;
            inputYAxis = yMoveAction.ReadValue<Vector2>().y;
        }

        void Update()
        {
            // If the debug menu is running, we don't want to conflict with its inputs.
            if (DebugManager.instance.displayRuntimeUI)
                return;

            UpdateInputs();

            if (inputChangeSpeed != 0.0f)
            {
                m_MoveSpeed += inputChangeSpeed * m_MoveSpeedIncrement;
                if (m_MoveSpeed < m_MoveSpeedIncrement) m_MoveSpeed = m_MoveSpeedIncrement;
            }

            bool moved = inputRotateAxisX != 0.0f || inputRotateAxisY != 0.0f || inputVertical != 0.0f || inputHorizontal != 0.0f || inputYAxis != 0.0f;
            if (moved)
            {
                float rotationX = transform.localEulerAngles.x;
                float newRotationY = transform.localEulerAngles.y + inputRotateAxisX;

                // Weird clamping code due to weird Euler angle mapping...
                float newRotationX = (rotationX - inputRotateAxisY);
                if (rotationX <= 90.0f && newRotationX >= 0.0f)
                    newRotationX = Mathf.Clamp(newRotationX, 0.0f, 90.0f);
                if (rotationX >= 270.0f)
                    newRotationX = Mathf.Clamp(newRotationX, 270.0f, 360.0f);

                transform.localRotation = Quaternion.Euler(newRotationX, newRotationY, transform.localEulerAngles.z);

                float moveSpeed = Time.deltaTime * m_MoveSpeed;
                if (fire1 || leftShiftBoost && leftShift)
                    moveSpeed *= m_Turbo;
                transform.position += transform.forward * (moveSpeed * inputVertical)
                                      + transform.right * (moveSpeed * inputHorizontal)
                                      + Vector3.up * (moveSpeed * inputYAxis);
            }
        }
    }
}
