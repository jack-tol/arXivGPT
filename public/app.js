document.addEventListener('DOMContentLoaded', function () {
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      // Look for all instances of the Copy button
      const copyButtons = document.querySelectorAll(
        '.MuiStack-root .MuiButtonBase-root[aria-label="Copy"]'
      );

      copyButtons.forEach(function (copyButton) {
        // Hide each Copy button found
        if (copyButton) {
          copyButton.style.display = 'none';
        }
      });
    });
  });

  // Start observing the chat container for changes (adjust the selector if needed)
  const chatContainer = document.querySelector('#root');
  if (chatContainer) {
    observer.observe(chatContainer, { childList: true, subtree: true });
  }
});

// change placeholder

window.addEventListener('load', () => {
  // Create a MutationObserver to watch for the textarea element being added to the DOM
  const observer = new MutationObserver((mutationsList, observer) => {
    const textArea = document.getElementById('chat-input');

    if (textArea) {
      // Once the textarea is found, update the placeholder
      textArea.placeholder = 'Message arXivGPT';
      console.log('Placeholder text updated successfully.');

      // Now set up another MutationObserver to watch for changes to the textarea's attributes
      const attributeObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (
            mutation.type === 'attributes' &&
            mutation.attributeName === 'placeholder'
          ) {
            // If placeholder changes, reset it to the desired value
            if (textArea.placeholder !== 'Message arXivGPT') {
              textArea.placeholder = 'Message arXivGPT';
              console.log('Placeholder text re-updated.');
            }
          }
        });
      });

      // Start observing the textarea for attribute changes
      attributeObserver.observe(textArea, {
        attributes: true, // Monitor attribute changes only
      });

      // Once we find the element and start observing it, disconnect the initial observer
      observer.disconnect();
    }
  });

  // Start observing the body for child element additions
  observer.observe(document.body, { childList: true, subtree: true });
});
