document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const submitButton = document.querySelector('button[type="submit"]');
    const resultImg = document.getElementById('result-img');
    const resultText = document.getElementById('result-text');
    
    try {
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';
        
        const formData = new FormData(this);
        
        const response = await fetch('/api/process/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Processing failed');
        }
        
        if (result.success) {
            if (result.image_base64) {
                resultImg.src = `data:image/png;base64,${result.image_base64}`;
                resultImg.style.display = 'block';
            } else {
                resultImg.style.display = 'none';
            }
            
            resultText.textContent = result.text || 'No text extracted';
        } else {
            throw new Error(result.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Processing error:', error);
        resultText.textContent = `Error: ${error.message}`;
        resultImg.style.display = 'none';
        
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Process Image';
    }
});
