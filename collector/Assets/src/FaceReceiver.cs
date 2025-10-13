using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using System.IO;

public class FaceReceiver : MonoBehaviour
{
    public int port = 4399;
    public GameObject faceMarker;
    public Vector3 positionOffset = Vector3.zero;
    public Vector3 rotationOffset = Vector3.zero;
    
    private UdpClient udpClient;
    private Thread receiveThread;
    private Vector3 latestPosition;
    private Quaternion latestRotation;
    private bool hasNewData = false;

    private List<ExportData> recordedData = new List<ExportData>();
    private bool isRecording = false;
    
    public bool IsRecording => isRecording;
    
    void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(ReceiveData);
        receiveThread.Start();
        Debug.Log($"[PC] Listening on port {port}");
        
        if (faceMarker != null && positionOffset == Vector3.zero)
        {
            positionOffset = faceMarker.transform.position;
            Debug.Log($"Using initial position as offset: {positionOffset}");
        }
    }
    
    void ReceiveData()
    {
        IPEndPoint endpoint = new IPEndPoint(IPAddress.Any, port);
        while (true)
        {
            try
            {
                byte[] data = udpClient.Receive(ref endpoint);
                string message = Encoding.UTF8.GetString(data);
                
                PoseData pose = JsonUtility.FromJson<PoseData>(message);
                latestPosition = new Vector3(pose.x, pose.y, -pose.z);
                latestRotation = new Quaternion(pose.qx, pose.qy, -pose.qz, -pose.qw);
                
                hasNewData = true;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Error: {e.Message}");
            }
        }
    }
    
    void Update()
    {
        if (hasNewData && faceMarker != null)
        {
            faceMarker.transform.position = latestPosition + positionOffset;
            faceMarker.transform.rotation = latestRotation * Quaternion.Euler(rotationOffset);
            hasNewData = false;
            
            if (isRecording)
            {
                Vector3 finalPos = latestPosition + positionOffset;
                recordedData.Add(new ExportData
                {
                    timestamp = Time.time,
                    x = finalPos.x,
                    y = finalPos.y,
                    z = finalPos.z
                });
            }
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            isRecording = !isRecording;
            Debug.Log(isRecording ? "Recording..." : "Stopped");
            
            if (!isRecording && recordedData.Count > 0)
            {
                SaveData();
            }
        }
        
        if (Input.GetKeyDown(KeyCode.O))
        {
            if (faceMarker != null)
            {
                positionOffset = faceMarker.transform.position - latestPosition;
                Debug.Log($"Offset updated: {positionOffset}");
            }
        }
    }
    
    void SaveData()
    {
        StringBuilder json = new StringBuilder();
        json.Append("[\n");
    
        for (int i = 0; i < recordedData.Count; i++)
        {
            var d = recordedData[i];
            json.Append($"  {{\"timestamp\": {d.timestamp}, \"x\": {d.x}, \"y\": {d.y}, \"z\": {d.z}}}");
            if (i < recordedData.Count - 1) json.Append(",");
            json.Append("\n");
        }
    
        json.Append("]");
    
        string folder = "Assets/";
        if (!Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }
    
        string filename = $"human_data_{System.DateTime.Now:yyyyMMdd_HHmmss}.json";
        string path = Path.Combine(folder, filename);
    
        File.WriteAllText(path, json.ToString());
        Debug.Log($"Saved {recordedData.Count} points to {path}");
        recordedData.Clear();
    }
    
    void OnDestroy()
    {
        receiveThread?.Abort();
        udpClient?.Close();
    }
}

[System.Serializable]
public struct PoseData
{
    public float x;
    public float y;
    public float z;
    public float qx;
    public float qy;
    public float qz;
    public float qw;
}

[System.Serializable]
public struct ExportData
{
    public float timestamp;
    public float x;
    public float y;
    public float z;
}