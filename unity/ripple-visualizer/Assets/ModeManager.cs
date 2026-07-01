using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ModeManager : MonoBehaviour
{
    public Button toggleModeButton;
    public GameObject editModePanel; // Reference to the edit mode panel
    private bool isEditMode = false;
    private string editModeText = "Edit Mode";
    private string playModeText = "Play Mode";
    private TextMeshProUGUI buttonText;

    // Start is called before the first frame update
    void Start()
    {
        buttonText = toggleModeButton.GetComponentInChildren<TextMeshProUGUI>();

        if (buttonText == null)
        {
            Debug.LogError("TextMeshProUGUI component not found on the button.");
            return;
        }

        toggleModeButton.onClick.AddListener(ToggleMode);
        UpdateButtonText();
        editModePanel.SetActive(false); // Ensure the panel is hidden initially
    }

    void ToggleMode()
    {
        isEditMode = !isEditMode;
        UpdateButtonText();

        if (isEditMode)
        {
            EnterEditMode();
        }
        else
        {
            SaveSettings();
            EnterPlayMode();
        }
    }

    void UpdateButtonText()
    {
        if (buttonText == null)
        {
            Debug.LogError("TextMeshProUGUI component not found on the button.");
            return;
        }

        buttonText.text = isEditMode ? playModeText : editModeText;
    }

    void EnterEditMode()
    {
        // Enable controls for editing
        Debug.Log("Entered Edit Mode");
        editModePanel.SetActive(true); // Show the edit mode panel
        // Example: Allow movement and modifications of elements
    }

    void EnterPlayMode()
    {
        // Disable edit controls and enter play mode
        Debug.Log("Entered Play Mode");
        editModePanel.SetActive(false); // Hide the edit mode panel
        // Example: Lock elements and allow the scene to be played
    }

    void SaveSettings()
    {
        // Save the current state of the elements
        Debug.Log("Settings Saved");
    }
}
