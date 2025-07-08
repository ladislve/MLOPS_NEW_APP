async function classify() {
  const text = document.getElementById('newsText').value;
  const resultDiv = document.getElementById('result');

  try {
    const response = await fetch('http://127.0.0.1:49885/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.text();
    resultDiv.innerText = `Prediction: ${result}`;
  } catch (err) {
    resultDiv.innerText = `Error: ${err.message}`;
  }
}
