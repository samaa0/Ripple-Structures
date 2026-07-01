using UnityEngine;

public class RoomColliderResizer : MonoBehaviour
{
    // Function to be called when you want to resize the collider
    public void ResizeCollider(float newWidth, float newHeight, float newDepth)
    {
        BoxCollider boxCollider = GetComponent<BoxCollider>();
        if (boxCollider != null)
        {
            boxCollider.size = new Vector3(newWidth, newHeight, newDepth);
        }
        else
        {
            Debug.LogError("BoxCollider component not found on the object.");
        }
    }
}