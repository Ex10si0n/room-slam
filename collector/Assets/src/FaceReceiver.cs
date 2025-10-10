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
    
    private List<FaceData> recordedData = new List<FaceData>();
    private bool isRecording = false;
    
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
                
                // Parse JSON from iOS app
                PoseData pose = JsonUtility.FromJson<PoseData>(message);
                
                // ARKit coordinate system to Unity conversion
                // ARKit: +X right, +Y up, +Z backward (right-handed)
                // Unity: +X right, +Y up, +Z forward (left-handed)
                latestPosition = new Vector3(pose.x, pose.y, -pose.z);
                
                // Convert quaternion: negate Z and W for coordinate system change
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
                recordedData.Add(new FaceData
                {
                    timestamp = Time.time,
                    position = latestPosition + positionOffset,
                    rotation = (latestRotation * Quaternion.Euler(rotationOffset)).eulerAngles
                });
            }
        }
        
        // Press R to toggle recording
        if (Input.GetKeyDown(KeyCode.R))
        {
            isRecording = !isRecording;
            Debug.Log(isRecording ? "Recording..." : "Stopped");
            
            if (!isRecording && recordedData.Count > 0)
            {
                SaveData();
            }
        }
        
        // Press O to update position offset
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
        string json = JsonUtility.ToJson(new FaceDataWrapper { data = recordedData }, true);
        string path = $"Assets/face_data_{System.DateTime.Now:yyyyMMdd_HHmmss}.json";
        File.WriteAllText(path, json);
        Debug.Log($"Saved {recordedData.Count} points to {path}");
        recordedData.Clear();
    }
    
    void OnDestroy()
    {
        receiveThread?.Abort();
        udpClient?.Close();
    }
}

// JSON structure from iOS app
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
public struct FaceData
{
    public float timestamp;
    public Vector3 position;
    public Vector3 rotation;
}

[System.Serializable]
public class FaceDataWrapper
{
    public List<FaceData> data;
}