using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class UDPSend2 : MonoBehaviour
{
    public string host = "127.0.0.1";
    public int port = 9000;

    private UdpClient udpClient;

    private void Start()
    {
        udpClient = new UdpClient();
    }

    // Call this method to send data
    public void SendLocation()
    {
        try
        {
            string locationData = SerializeHierarchy();
            byte[] bytesToSend = Encoding.UTF8.GetBytes(locationData);
            udpClient.Send(bytesToSend, bytesToSend.Length, host, port);
            Debug.Log("Location data sent!");
        }
        catch (System.Exception err)
        {
            Debug.LogError(err.ToString());
        }
    }

    private string SerializeHierarchy()
    {
        Transform myRoom = GameObject.Find("My Room").transform;
        Dictionary<string, List<string>> locationData = new Dictionary<string, List<string>>();

        foreach (Transform child in myRoom)
        {
            if (!locationData.ContainsKey("My Room"))
            {
                locationData["My Room"] = new List<string>();
            }
            locationData["My Room"].Add(child.gameObject.name);
        }

        return JsonUtility.ToJson(locationData);
    }

    private void OnApplicationQuit()
    {
        udpClient.Close();
    }
}