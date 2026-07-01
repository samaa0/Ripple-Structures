using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateCamera : MonoBehaviour
{
    public float panSpeed = 0.1f;
    public float rotationSpeed = 10.0f;
    public float zoomSpeed = 10f;
    private float X;
    private float Y;

    void Update()
    {
        // Rotate camera
        if (Input.GetMouseButton(0)) // Left click
        {
            X += Input.GetAxis("Mouse X") * rotationSpeed;
            Y -= Input.GetAxis("Mouse Y") * rotationSpeed;
            Y = Mathf.Clamp(Y, -80, 80);

            transform.rotation = Quaternion.Euler(Y, X, 0);
        }

        // Pan camera
        if (Input.GetMouseButton(2)) // Middle click
        {
            Vector3 delta = new Vector3(-Input.GetAxis("Mouse X"), -Input.GetAxis("Mouse Y"), 0);
            transform.position += delta * panSpeed;
        }

        // Zoom camera
        float scrollData;
        scrollData = Input.GetAxis("Mouse ScrollWheel");
        transform.position = transform.position - transform.forward * scrollData * zoomSpeed * Time.deltaTime;
    }
}