function removeCopyButtons() {
  document.querySelectorAll('button[data-state="closed"]').forEach((button) => {
    if (button.querySelector('.lucide-copy')) {
      button.remove();
    }
  });
}

function updatePlaceholder() {
  document.querySelectorAll('#chat-input').forEach((input) => {
    input.setAttribute('data-placeholder', 'Message arXivGPT');

    if (input.textContent.trim() === '') {
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
}

removeCopyButtons();
updatePlaceholder();

const observer = new MutationObserver(() => {
  removeCopyButtons();
  updatePlaceholder();
});

observer.observe(document.body, { childList: true, subtree: true });
