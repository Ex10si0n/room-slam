using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;

public class DebugConsole : MonoBehaviour
{
    [Header("UI References")]
    public GameObject consolePanel;
    public TMP_Text consoleText;
    public ScrollRect scrollRect;
    
    [Header("Settings")]
    public int maxLines = 20;
    public bool showTimestamp = true;
    public KeyCode toggleKey = KeyCode.BackQuote;
    public float autoScrollThreshold = 0.1f;
    
    private Queue<string> logQueue = new Queue<string>();
    private bool isVisible = false;
    private bool needScrollToBottom = false;
    private bool userIsScrolling = false;
    
    void OnEnable()
    {
        Application.logMessageReceived += HandleLog;
        
        if (scrollRect != null)
        {
            scrollRect.onValueChanged.AddListener(OnScrollChanged);
        }
    }
    
    void OnDisable()
    {
        Application.logMessageReceived -= HandleLog;
        
        if (scrollRect != null)
        {
            scrollRect.onValueChanged.RemoveListener(OnScrollChanged);
        }
    }
    
    void Start()
    {
        if (consolePanel != null)
        {
            consolePanel.SetActive(isVisible);
        }
    }
    
    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            isVisible = !isVisible;
            if (consolePanel != null)
            {
                consolePanel.SetActive(isVisible);
            }
        }
    }
    
    void LateUpdate()
    {
        if (needScrollToBottom && !userIsScrolling)
        {
            if (scrollRect != null && scrollRect.verticalNormalizedPosition <= autoScrollThreshold)
            {
                ScrollToBottom();
            }
            needScrollToBottom = false;
        }
    }
    
    void OnScrollChanged(Vector2 scrollPos)
    {
        if (scrollRect != null)
        {
            userIsScrolling = scrollRect.verticalNormalizedPosition > autoScrollThreshold;
        }
    }
    
    void HandleLog(string logString, string stackTrace, LogType type)
    {
        string prefix = "";
        
        switch (type)
        {
            case LogType.Error:
            case LogType.Exception:
                prefix = "<color=red>[ERROR]</color> ";
                break;
            case LogType.Warning:
                prefix = "<color=yellow>[WARN]</color> ";
                break;
            default:
                prefix = "<color=green>[INFO]</color> ";
                break;
        }
        
        string timestamp = showTimestamp ? $"[{Time.time:F2}] " : "";
        string formattedLog = $"{timestamp}{prefix}{logString}";
        
        logQueue.Enqueue(formattedLog);
        
        while (logQueue.Count > maxLines)
        {
            logQueue.Dequeue();
        }
        
        UpdateConsoleText();
    }
    
    void UpdateConsoleText()
    {
        if (consoleText != null)
        {
            consoleText.text = string.Join("\n", logQueue);
            needScrollToBottom = true;
        }
    }
    
    void ScrollToBottom()
    {
        if (scrollRect != null)
        {
            Canvas.ForceUpdateCanvases();
            scrollRect.verticalNormalizedPosition = 0f;
        }
    }
    
    public void AddLog(string message)
    {
        Debug.Log(message);
    }
    
    public void Clear()
    {
        logQueue.Clear();
        UpdateConsoleText();
    }
}