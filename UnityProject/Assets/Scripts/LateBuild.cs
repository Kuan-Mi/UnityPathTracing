using UnityEngine;

namespace DefaultNamespace
{
    public class LateBuild : MonoBehaviour
    {
        private async void Start()
        {
            await Awaitable.WaitForSecondsAsync(1);
            
            PathTracingDataBuilder.instance.Build();
        }
        
    }
}