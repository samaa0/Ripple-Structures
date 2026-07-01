using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.Text;

public class RXCollisionHandler : MonoBehaviour
{
    public float debounceTime = 1.0f; // Time to prevent multiple triggers
    public string serverHost = "127.0.0.1";
    public int serverPort = 12345;
    private bool isDebouncingEnter = false;
    private bool isDebouncingExit = false;
    private UdpClient udpClient; // Separate UDP client for sending enter/exit messages

    private void Start()
    {
        // Initialize the UDP client
        udpClient = new UdpClient();
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Room") && !isDebouncingEnter)
        {
            StartCoroutine(DebounceEnter(other));
        }
    }

    private IEnumerator DebounceEnter(Collider other)
    {
        isDebouncingEnter = true;
        yield return new WaitForSeconds(debounceTime);
        if (other != null && other.CompareTag("Room"))
        {
            Debug.Log(gameObject.name + " entered " + other.gameObject.name);
            // Send the entered message using the separate UDP client
            SendUDPMessage($"{gameObject.name} entered: {other.gameObject.name}");
        }
        isDebouncingEnter = false;
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Room") && !isDebouncingExit)
        {
            StartCoroutine(DebounceExit(other));
        }
    }

    private IEnumerator DebounceExit(Collider other)
    {
        isDebouncingExit = true;
        yield return new WaitForSeconds(debounceTime);
        if (other != null && other.CompareTag("Room"))
        {
            Debug.Log(gameObject.name + " exited " + other.gameObject.name);
            // Send the exited message using the separate UDP client
            SendUDPMessage($"{gameObject.name} exited: {other.gameObject.name}");
        }
        isDebouncingExit = false;
    }

    private void SendUDPMessage(string message)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            udpClient.Send(data, data.Length, serverHost, serverPort); // Send to the Python server
            Debug.Log("Sent UDP message: " + message);
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.Message);
        }
    }

    private void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}
