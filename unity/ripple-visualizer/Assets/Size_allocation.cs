using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class SyncColliderWithScale : MonoBehaviour
{
    private BoxCollider _boxCollider;
    private Vector3 _previousScale;

    void Start()
    {
        _boxCollider = GetComponent<BoxCollider>();
        _previousScale = transform.localScale;
        UpdateColliderSize();
    }

    void Update()
    {
        // Check if the local scale has changed since the last frame
        if (_previousScale != transform.localScale)
        {
            UpdateColliderSize();
            _previousScale = transform.localScale;
        }
    }

    private void UpdateColliderSize()
    {
        // Assuming the cube is 1x1x1 in size, the collider size should match the local scale
        // If your cube is not 1x1x1, you'll need to adjust the size based on the actual size of your mesh
        _boxCollider.size = Vector3.one;
        _boxCollider.size = new Vector3(transform.localScale.x / transform.lossyScale.x,
                                        transform.localScale.y / transform.lossyScale.y,
                                        transform.localScale.z / transform.lossyScale.z);
    }
}