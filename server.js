const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/classify', (req, res) => {
  const { url } = req.body;
  
  // Call Python script for text classification
  const python = spawn('python', ['classify_text.py', url]);
  
  python.stdout.on('data', (data) => {
    const result = JSON.parse(data.toString());
    res.json(result);
  });
});

app.post('/classify-image', (req, res) => {
  const { imageUrl } = req.body;
  
  // Call Python script for image classification
  const python = spawn('python', ['classify_image.py', imageUrl]);
  
  python.stdout.on('data', (data) => {
    const result = JSON.parse(data.toString());
    res.json(result);
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

