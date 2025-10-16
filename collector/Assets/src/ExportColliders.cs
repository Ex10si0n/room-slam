using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class ColliderExporter : MonoBehaviour
{
    public GameObject targetObject;
    public string exportFileName = "colliders.json";

    [System.Serializable]
    public class ColliderData
    {
        public string type;
        public Vector3 center;
        public Vector3 size;      // For BoxCollider
        public float radius;      // For Sphere/Capsule
        public float height;      // For Capsule
    }

    [System.Serializable]
    public class ColliderList
    {
        public List<ColliderData> colliders = new List<ColliderData>();
    }

    void Start()
    {
        ExportColliders();
    }

    public void ExportColliders()
    {
        if (targetObject == null)
        {
            Debug.LogError("Target object not assigned.");
            return;
        }

        Collider[] colliders = targetObject.GetComponentsInChildren<Collider>();
        ColliderList list = new ColliderList();

        foreach (Collider col in colliders)
        {
            ColliderData data = new ColliderData
            {
                type = col.GetType().Name,
                center = col.bounds.center
            };

            if (col is BoxCollider box)
            {
                data.size = box.size;
            }
            else if (col is SphereCollider sphere)
            {
                data.radius = sphere.radius;
            }
            else if (col is CapsuleCollider capsule)
            {
                data.radius = capsule.radius;
                data.height = capsule.height;
            }

            list.colliders.Add(data);
        }

        string json = JsonUtility.ToJson(list, true);
        string folder = "Assets/";
        if (!Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }
    
        string filename = $"colliders.json";
        string path = Path.Combine(folder, filename);
    
        File.WriteAllText(path, json.ToString());
        Debug.Log($"Saved colliders");
    }
}