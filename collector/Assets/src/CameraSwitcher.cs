using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera thirdPersonCam;
    public Camera topDownCam;
    public KeyCode toggleKey = KeyCode.V;

    void Start()
    {
        SetActive(thirdPerson: true);
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            bool thirdOn = thirdPersonCam.enabled;
            SetActive(!thirdOn);
        }
    }

    void SetActive(bool thirdPerson)
    {
        if (thirdPersonCam) thirdPersonCam.enabled = thirdPerson;
        if (topDownCam)     topDownCam.enabled     = !thirdPerson;

        if (thirdPersonCam) thirdPersonCam.tag = thirdPerson ? "MainCamera" : "Untagged";
        if (topDownCam)     topDownCam.tag     = thirdPerson ? "Untagged"   : "MainCamera";
    }
}