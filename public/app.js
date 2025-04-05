function removeCopyButtons() {
  document.querySelectorAll('button[data-state="closed"]').forEach((button) => {
    if (button.querySelector('.lucide-copy')) {
      button.remove();
    }
  });
}

removeCopyButtons();

const observer = new MutationObserver(() => {
  removeCopyButtons();
});

observer.observe(document.body, { childList: true, subtree: true });
