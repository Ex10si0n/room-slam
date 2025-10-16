using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using StarterAssets;

public class RobotAI : MonoBehaviour
{
    [Header("Movement Settings")]
    public float aiMoveSpeed = 1f;
    public float rotateSpeed = 90f;
    public float changeDirectionInterval = 5f;
    
    [Header("Movement Mode")]
    public MovementMode mode = MovementMode.RandomWalk;
    public Transform[] waypoints;
    
    [Header("Input Detection")]
    public float inputTimeoutDuration = 0.5f;
    
    private StarterAssetsInputs starterAssetsInputs;
    private CharacterController characterController;
    
    public bool IsRecording => isRecording;
    
    private Vector2 aiMoveInput;
    private float nextDirectionChange;
    private int currentWaypointIndex = 0;
    
    private bool isPlayerControlling = false;
    private bool isForcedStop = false; // Force stop flag
    private float lastInputTime = 0f;
    
    private List<RobotData> recordedData = new List<RobotData>();
    private bool isRecording = false;
    
    void Start()
    {
        starterAssetsInputs = GetComponent<StarterAssetsInputs>();
        characterController = GetComponent<CharacterController>();
        
        if (starterAssetsInputs == null)
        {
            Debug.LogError("[Robot] StarterAssetsInputs not found!");
        }
        
        ChangeDirection();
    }
    
    void Update()
    {
        // Press F to toggle force stop
        if (Input.GetKeyDown(KeyCode.F))
        {
            isForcedStop = !isForcedStop;
            Debug.Log(isForcedStop ? "[Robot] Force STOPPED" : "[Robot] Force stop RELEASED");
            
            // Clear input immediately when forced to stop
            if (isForcedStop && starterAssetsInputs != null)
            {
                starterAssetsInputs.MoveInput(Vector2.zero);
            }
        }
        
        CheckPlayerInput();
        
        // If forced to stop, prevent all movement
        if (isForcedStop)
        {
            if (starterAssetsInputs != null)
            {
                starterAssetsInputs.MoveInput(Vector2.zero);
            }
        }
        else if (!isPlayerControlling)
        {
            // AI control
            switch (mode)
            {
                case MovementMode.RandomWalk:
                    RandomWalkInput();
                    break;
                case MovementMode.Patrol:
                    PatrolInput();
                    break;
                case MovementMode.CircularPath:
                    CircularPathInput();
                    break;
            }

            if (starterAssetsInputs != null)
            {
                starterAssetsInputs.MoveInput(aiMoveInput);
            }
        }

        // Record position if recording is enabled
        if (isRecording)
        {
            RecordPosition();
        }

        // Press T to toggle recording
        if (Input.GetKeyDown(KeyCode.T))
        {
            isRecording = !isRecording;
            Debug.Log(isRecording ? "[Robot] Recording..." : "[Robot] Stopped");
            
            if (!isRecording && recordedData.Count > 0)
            {
                SaveData();
            }
        }
    }
    
    void CheckPlayerInput()
    {
        if (starterAssetsInputs == null) return;
        
        // Detect real player input
        bool hasRealInput = Input.GetAxis("Horizontal") != 0 ||
                            Input.GetAxis("Vertical") != 0 ||
                            Input.GetKey(KeyCode.W) ||
                            Input.GetKey(KeyCode.A) ||
                            Input.GetKey(KeyCode.S) ||
                            Input.GetKey(KeyCode.D) ||
                            starterAssetsInputs.jump ||
                            starterAssetsInputs.sprint;
        
        if (hasRealInput)
        {
            lastInputTime = Time.time;
            
            if (!isPlayerControlling)
            {
                isPlayerControlling = true;
                isForcedStop = false; // Auto release force stop when player takes control
                Debug.Log("[Robot] Player control enabled");
            }
        }
        else
        {
            // Switch back to AI control after timeout
            if (isPlayerControlling && Time.time - lastInputTime > inputTimeoutDuration)
            {
                isPlayerControlling = false;
                Debug.Log("[Robot] AI control enabled");
            }
        }
    }
    
    void RandomWalkInput()
    {
        // Change direction periodically
        if (Time.time > nextDirectionChange)
        {
            ChangeDirection();
        }
        
        aiMoveInput = new Vector2(aiMoveInput.x, aiMoveInput.y) * aiMoveSpeed;
    }
    
    void PatrolInput()
    {
        if (waypoints == null || waypoints.Length == 0)
        {
            aiMoveInput = Vector2.zero;
            return;
        }
        
        Transform target = waypoints[currentWaypointIndex];
        Vector3 direction = (target.position - transform.position);
        direction.y = 0;
        direction.Normalize();
        
        // Convert to local space relative to character facing
        Vector3 localDir = transform.InverseTransformDirection(direction);
        aiMoveInput = new Vector2(localDir.x, localDir.z) * aiMoveSpeed;
        
        // Switch to next waypoint when reached
        if (Vector3.Distance(transform.position, target.position) < 1f)
        {
            currentWaypointIndex = (currentWaypointIndex + 1) % waypoints.Length;
        }
    }
    
    void CircularPathInput()
    {
        float angularSpeed = 30f;
        float angle = Time.time * angularSpeed * Mathf.Deg2Rad;
        
        // Set rotation
        transform.rotation = Quaternion.Euler(0, angle * Mathf.Rad2Deg, 0);
        
        // Move forward
        aiMoveInput = new Vector2(0, 1) * aiMoveSpeed;
    }
    
    void OnControllerColliderHit(ControllerColliderHit hit)
    {
        // Bounce off obstacles when AI is in control and not forced to stop
        if (!isPlayerControlling && !isForcedStop)
        {
            Vector3 hitNormal = hit.normal;
            hitNormal.y = 0;
            
            // Calculate reflection direction
            Vector3 currentDir = new Vector3(aiMoveInput.x, 0, aiMoveInput.y);
            Vector3 reflectDir = Vector3.Reflect(currentDir, hitNormal).normalized;
            
            aiMoveInput = new Vector2(reflectDir.x, reflectDir.z);
            nextDirectionChange = Time.time + changeDirectionInterval;
        }
    }
    
    void ChangeDirection()
    {
        // Generate random direction
        float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
        aiMoveInput = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
        
        nextDirectionChange = Time.time + changeDirectionInterval;
    }
    
    void RecordPosition()
    {
        recordedData.Add(new RobotData
        {
            timestamp = Time.time,
            x = transform.position.x,
            y = transform.position.y,
            z = transform.position.z
        });
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
    
        string filename = $"agent_data_{System.DateTime.Now:yyyyMMdd_HHmmss}.json";
        string path = Path.Combine(folder, filename);
    
        File.WriteAllText(path, json.ToString());
        Debug.Log($"[Robot] Saved {recordedData.Count} points to {path}");
        recordedData.Clear();
    }
}

public enum MovementMode
{
    RandomWalk,
    Patrol,
    CircularPath
}

[System.Serializable]
public struct RobotData
{
    public float timestamp;
    public float x;
    public float y;
    public float z;
}