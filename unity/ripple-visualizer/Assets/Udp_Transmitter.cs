using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System;

public class UDPSend : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 12345;
    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    void Start()
    {
        // Set up the UDP client to send to the local machine
        udpClient = new UdpClient();
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(host), port);

        // Serialize the scene structure and send
        string sceneJson = SerializeSceneStructure();
        if (!string.IsNullOrEmpty(sceneJson))
        {
            SendData(sceneJson);
        }
    }

    string SerializeSceneStructure()
    {
        // Create a dictionary to represent the scene hierarchy
        Dictionary<string, List<string>> sceneHierarchy = new Dictionary<string, List<string>>();

        // Find the "Home" GameObject in the scene
        GameObject homeGameObject = GameObject.Find("Home");
        if (homeGameObject != null)
        {
            Transform homeTransform = homeGameObject.transform;

            // Iterate over each room (child of "Home")
            foreach (Transform roomTransform in homeTransform)
            {
                List<string> items = new List<string>();

                // Iterate over each item (child of the room)
                foreach (Transform itemTransform in roomTransform)
                {
                    // Serialize each child's name in the room
                    items.Add(itemTransform.name);
                }

                // Add the room and its items to the dictionary
                sceneHierarchy[roomTransform.name] = items;
            }
        }
        else
        {
            Debug.LogError("Unable to find a GameObject named 'Home'. Please make sure it exists and is active in the scene.");
            return null;
        }

        // Convert the hierarchy to a JSON string
        SerializationWrapper wrapper = new SerializationWrapper(sceneHierarchy);
        string json = JsonUtility.ToJson(wrapper, true);
        return json;
    }

    public void SendData(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            udpClient.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            Debug.LogError(err.ToString());
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }

    // Wrapper class for serialization since Unity's JsonUtility does not support serializing Dictionary directly
    [Serializable]
    private class SerializationWrapper
    {
        public List<RoomData> rooms;

        public SerializationWrapper(Dictionary<string, List<string>> target)
        {
            rooms = new List<RoomData>();
            foreach (KeyValuePair<string, List<string>> entry in target)
            {
                RoomData room = new RoomData
                {
                    roomName = entry.Key,
                    items = new List<ItemData>()
                };

                foreach (string itemName in entry.Value)
                {
                    room.items.Add(new ItemData { itemName = itemName });
                }

                rooms.Add(room);
            }
        }
    }

    [Serializable]
    private class RoomData
    {
        public string roomName;
        public List<ItemData> items;
    }

    [Serializable]
    private class ItemData
    {
        public string itemName;
    }
}
