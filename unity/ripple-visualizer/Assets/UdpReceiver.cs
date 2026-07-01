// using System;
// using System.Net;
// using System.Net.Sockets;
// using System.Text;
// using System.Threading;
// using UnityEngine;
// using System.Collections.Generic;

// public class UdpReceiver : MonoBehaviour
// {
//     Thread receiveThread;
//     UdpClient client;
//     public int port = 5065; // Port number should match the one used by Python script
    
//     private Dictionary<string, GameObject> rxObjects = new Dictionary<string, GameObject>();
//     private Dictionary<string, Color> rxColors = new Dictionary<string, Color>();
//     private Dictionary<string, float> rxLastUpdateTime = new Dictionary<string, float>(); // Track when each RX was last updated

//     // Start is called before the first frame update
//     void Start()
//     {
//         receiveThread = new Thread(new ThreadStart(ReceiveData));
//         receiveThread.IsBackground = true;
//         receiveThread.Start();
//     }

//     // This method is called when the MonoBehaviour will be destroyed
//     void OnDestroy()
//     {
//         if (receiveThread != null && receiveThread.IsAlive)
//         {
//             receiveThread.Abort();
//         }
//         client.Close();
//     }

//     // Receive data, parse it, and update/create objects
//     private void ReceiveData()
//     {
//         client = new UdpClient(port);
//         while (true)
//         {
//             try
//             {
//                 IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
//                 byte[] data = client.Receive(ref anyIP);
//                 string text = Encoding.UTF8.GetString(data);
//                 // Debug.Log(text);
//                 UpdateObjects(text);
//             }
//             catch (Exception err)
//             {
//                 print(err.ToString());
//             }
//         }
//     }

// private void UpdateObjects(string data)
// {
//     UnityMainThreadDispatcher.Instance().Enqueue(() =>
//     {
//         string[] parts = data.Split(',');

//         if (parts.Length < 5)
//         {
//             Debug.LogError("Received data does not have enough parts: " + data);
//             return;
//         }

//         string rxId = "RX[" + parts[0] + "]";
//         Vector3 position = new Vector3(
//             float.Parse(parts[1]),
//             float.Parse(parts[3]), // Y and Z are swapped in Unity compared to the coordinate system used by Python
//             float.Parse(parts[2]));

//         if (!rxColors.ContainsKey(rxId))
//         {
//             rxColors[rxId] = GetUniqueColor(rxId);
//         }

//         if (rxObjects.ContainsKey(rxId))
//         {
//             rxObjects[rxId].transform.position = position;
//             rxObjects[rxId].GetComponent<Renderer>().material.color = rxColors[rxId];
//         }
//         else
//         {
//             GameObject rxObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
//             rxObject.name = "User " + parts[0]; // Naming the sphere based on RX number
//             rxObject.transform.position = position;
//             rxObject.GetComponent<Renderer>().material.color = rxColors[rxId];
//             rxObject.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f); // Set the scale to half the original size

//             // Add Rigidbody and set it to kinematic
//             Rigidbody rb = rxObject.AddComponent<Rigidbody>();
//             rb.isKinematic = true;

//             // Add a trigger collider
//             SphereCollider sc = rxObject.GetComponent<SphereCollider>();
//             sc.isTrigger = true;

//             // Assign RX object to the dictionary
//             rxObjects.Add(rxId, rxObject);

//             // Add the RXCollisionHandler script to handle trigger events
//             rxObject.AddComponent<RXCollisionHandler>();
//         }

//         // Update the last update time for this RX
//         rxLastUpdateTime[rxId] = Time.time;
//     });
// }


//     // This method generates a unique color based on the RX ID
//     private Color GetUniqueColor(string id)
//     {
//         System.Random random = new System.Random(id.GetHashCode());
//         return new Color(
//             (float)random.NextDouble(),
//             (float)random.NextDouble(),
//             (float)random.NextDouble()
//         );
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         List<string> keysToRemove = new List<string>();

//         // Check for RXs that have not been updated
//         foreach (var rx in rxLastUpdateTime)
//         {
//             if (Time.time - rx.Value > 2f) // 2 seconds have passed
//             {
//                 keysToRemove.Add(rx.Key);
//             }
//         }

//         // Remove any RXs that are outdated
//         foreach (var key in keysToRemove)
//         {
//             if (rxObjects.ContainsKey(key))
//             {
//                 Destroy(rxObjects[key]); // Destroy the GameObject
//                 rxObjects.Remove(key);   // Remove from the dictionary
//             }

//             rxLastUpdateTime.Remove(key); // Remove the timestamp from the dictionary
//         }
//     }
// }

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections.Generic;

public class UdpReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5065; // Port number should match the one used by Python script

    private Dictionary<string, GameObject> rxObjects = new Dictionary<string, GameObject>();
    private Dictionary<string, Color> rxColors = new Dictionary<string, Color>();
    private Dictionary<string, float> rxLastUpdateTime = new Dictionary<string, float>(); // Track when each RX was last updated

    // Start is called before the first frame update
    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    // This method is called when the MonoBehaviour will be destroyed
    void OnDestroy()
    {
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }
        client.Close();
    }

    // Receive data, parse it, and update/create objects
    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                UpdateObjects(text);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    private void UpdateObjects(string data)
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            string[] parts = data.Split(',');

            if (parts.Length < 4)
            {
                Debug.LogError("Received data does not have enough parts: " + data);
                return;
            }

            string rxId = "RX[" + parts[0] + "]";
            Vector3 position = new Vector3(
                float.Parse(parts[1]),
                float.Parse(parts[3]), // Y and Z are swapped in Unity compared to the coordinate system used by Python
                float.Parse(parts[2]));

            if (!rxColors.ContainsKey(rxId))
            {
                rxColors[rxId] = GetUniqueColor(rxId);
            }

            if (rxObjects.ContainsKey(rxId))
            {
                rxObjects[rxId].transform.position = position;
                rxObjects[rxId].GetComponent<Renderer>().material.color = rxColors[rxId];
            }
            else
            {
                GameObject rxObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                rxObject.name = "User " + parts[0]; // Naming the sphere based on RX number
                rxObject.transform.position = position;
                rxObject.GetComponent<Renderer>().material.color = rxColors[rxId];
                rxObject.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f); // Set the scale to half the original size

                // Add Rigidbody and set it to kinematic
                Rigidbody rb = rxObject.AddComponent<Rigidbody>();
                rb.isKinematic = true;

                // Add a trigger collider
                SphereCollider sc = rxObject.GetComponent<SphereCollider>();
                sc.isTrigger = true;

                // Assign RX object to the dictionary
                rxObjects.Add(rxId, rxObject);

                // Add the RXCollisionHandler script to handle trigger events
                rxObject.AddComponent<RXCollisionHandler>();
            }

            // Update the last update time for this RX
            rxLastUpdateTime[rxId] = Time.time;
        });
    }

    // This method generates a unique color based on the RX ID
    private Color GetUniqueColor(string id)
    {
        System.Random random = new System.Random(id.GetHashCode());
        return new Color(
            (float)random.NextDouble(),
            (float)random.NextDouble(),
            (float)random.NextDouble()
        );
    }

    // Update is called once per frame
    void Update()
    {
        List<string> keysToRemove = new List<string>();

        // Check for RXs that have not been updated
        foreach (var rx in rxLastUpdateTime)
        {
            if (Time.time - rx.Value > 2f) // 2 seconds have passed
            {
                keysToRemove.Add(rx.Key);
            }
        }

        // Remove any RXs that are outdated
        foreach (var key in keysToRemove)
        {
            if (rxObjects.ContainsKey(key))
            {
                Destroy(rxObjects[key]); // Destroy the GameObject
                rxObjects.Remove(key);   // Remove from the dictionary
            }

            rxLastUpdateTime.Remove(key); // Remove the timestamp from the dictionary
        }
    }
}
