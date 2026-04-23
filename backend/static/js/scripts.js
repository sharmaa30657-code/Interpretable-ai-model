const previewImage = (event) => {
  const file = event.target.files[0];
  const preview = document.getElementById('preview');
  const previewContainer = document.querySelector('.file-preview');
  const uploadLabel = document.querySelector('.upload-instructions strong');

  if (!file) {
    preview.style.display = 'none';
    uploadLabel.textContent = 'Click to upload an image';
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
    uploadLabel.textContent = file.name;
  };
  reader.readAsDataURL(file);
};

const form = document.getElementById('uploadForm');
if (form) {
  form.addEventListener('submit', (event) => {
    const loader = document.getElementById('loader');
    loader.classList.add('visible');
  });
}

window.previewImage = previewImage;
